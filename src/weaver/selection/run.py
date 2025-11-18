from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


@dataclass
class FeatureStats:
    """Running statistics for a single verifier column."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def update(self, value: float) -> None:
        """Welford online update."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / max(self.count, 1)
        delta2 = value - self.mean
        self.m2 += delta * delta2
        if self.min_value is None or value < self.min_value:
            self.min_value = value
        if self.max_value is None or value > self.max_value:
            self.max_value = value

    @property
    def variance(self) -> float:
        if self.count <= 0:
            return 0.0
        return max(self.m2 / self.count, 0.0)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


def _sigmoid(value: float) -> float:
    """Numerically stable logistic."""
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class WeaverScorer:
    """
    Minimal, dependency-free approximation of Weaver's stage-2 scorer.
    """

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        baseline: float = 0.5,
        min_std: float = 1e-3,
        method: str = "auto",
        prior: float = 0.5,
        temperature: float = 1.0,
        use_zscore: bool = False,
    ) -> None:
        if not rows:
            raise ValueError("WeaverScorer requires at least one feature row.")
        if not (0.0 < prior < 1.0):
            raise ValueError("prior must be in (0, 1)")
        self.rows = list(rows)
        self.baseline = baseline
        self.min_std = max(min_std, 1e-6)
        self.method = method
        self.prior = prior
        self.temperature = max(temperature, 1e-6)
        self.use_zscore = use_zscore

        self.feature_names = sorted(
            {key for row in self.rows for key in ((row.get("features") or {}).keys())}
        )
        if not self.feature_names:
            self.weights: list[float] = []
            self.stats: Dict[str, FeatureStats] = {}
        else:
            self.stats = {name: FeatureStats() for name in self.feature_names}
            self._populate_stats()
            self.weights = self._compute_weights()

        self.bias = math.log(self.prior / (1.0 - self.prior))

    def _populate_stats(self) -> None:
        for row in self.rows:
            feats = row.get("features") or {}
            for name in self.feature_names:
                value = float(feats.get(name, self.baseline))
                self.stats[name].update(value)

    def _compute_weights(self) -> list[float]:
        weights: list[float] = []
        for name in self.feature_names:
            stats = self.stats[name]
            std = max(stats.std, self.min_std)
            if self.method == "uniform":
                weight = 1.0
            elif self.method == "variance":
                spread = (
                    (stats.max_value - stats.min_value)
                    if stats.max_value is not None and stats.min_value is not None
                    else 0.0
                )
                weight = spread if spread else stats.variance
            else:  # auto
                spread = max(stats.variance, std, self.min_std)
                direction = 1.0 if stats.mean >= self.baseline else -1.0
                weight = direction * spread
            weights.append(weight)

        norm = sum(abs(w) for w in weights)
        if norm <= 0.0:
            return [0.0 for _ in weights]
        return [w / norm for w in weights]

    def _transform_value(self, name: str, value: float) -> float:
        stats = self.stats[name]
        if self.use_zscore:
            std = max(stats.std, self.min_std)
            return (value - stats.mean) / std
        return value

    def _vectorise_row(self, row: Mapping[str, Any]) -> list[float]:
        feats = row.get("features") or {}
        vector: list[float] = []
        for name in self.feature_names:
            raw = float(feats.get(name, self.baseline))
            vector.append(self._transform_value(name, raw))
        return vector

    def _score_vector(self, vector: Sequence[float]) -> float:
        total = self.bias
        for weight, value in zip(self.weights, vector):
            total += weight * value
        total /= self.temperature
        return _sigmoid(total)

    def score_all(self) -> list[float]:
        if not self.feature_names:
            return [self.prior for _ in self.rows]
        vectors = [self._vectorise_row(row) for row in self.rows]
        return [self._score_vector(vec) for vec in vectors]


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline: float = 0.5,
    min_std: float = 1e-3,
    method: str = "auto",
    prior: float = 0.5,
    temperature: float = 1.0,
    use_zscore: bool = False,
) -> list[float]:
    scorer = WeaverScorer(
        rows,
        baseline=baseline,
        min_std=min_std,
        method=method,
        prior=prior,
        temperature=temperature,
        use_zscore=use_zscore,
    )
    return scorer.score_all()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:  # pragma: no cover
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
    if not rows:
        raise ValueError(f"{path} did not contain any JSONL rows")
    return rows


def _write_scores(path: Path, scores: Sequence[float]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for score in scores:
            handle.write(json.dumps({"score": score}) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m weaver.selection.run",
        description="Lightweight Weaver scorer that consumes Forge JSONL batches.",
    )
    parser.add_argument("--input", type=Path, required=True, help="JSONL file with Forge rows.")
    parser.add_argument("--output", type=Path, required=True, help="Where to store the scores JSONL.")
    parser.add_argument("--baseline", type=float, default=0.5, help="Fallback for missing verifier values.")
    parser.add_argument("--min-std", type=float, default=1e-3, help="Numerical floor for std when normalising.")
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "uniform", "variance"],
        help="Strategy for deriving feature weights.",
    )
    parser.add_argument("--prior", type=float, default=0.5, help="Prior probability fed into the sigmoid link.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature applied before sigmoid.")
    parser.add_argument("--zscore", action="store_true", help="Z-normalise verifier columns before scoring.")
    parser.add_argument("--seed", type=int, default=0, help="Compatibility flag (unused).")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    rows = _load_rows(args.input)
    scores = score_rows(
        rows,
        baseline=args.baseline,
        min_std=args.min_std,
        method=args.method,
        prior=args.prior,
        temperature=args.temperature,
        use_zscore=args.zscore,
    )
    _write_scores(args.output, scores)


if __name__ == "__main__":  # pragma: no cover
    main()

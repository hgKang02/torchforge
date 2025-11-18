from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .hf_rm import HFRewardModel


@dataclass
class VerifierEnsembleReward:
    """
    Wrap a list of HF reward models and optionally run Weaver selection
    over their raw scores. Callable signature mirrors the builtin rewards.
    """

    rm_specs: List[Dict[str, Any]]
    ensemble_reduce: str = "mean"
    use_weaver: bool = False
    weaver_cfg: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.rm_specs:
            raise ValueError("VerifierEnsembleReward requires at least one rm_spec.")

        self._rm_models: List[HFRewardModel] = []
        self._rm_names: List[str] = []
        sentinel = object()
        for idx, spec in enumerate(self.rm_specs):
            spec_copy = dict(spec)
            name = spec_copy.pop("name", None) or spec_copy.get("model_id") or f"rm_{idx}"
            self._rm_names.append(name)
            dtype = spec_copy.get("torch_dtype", sentinel)
            if dtype is not sentinel:
                spec_copy["torch_dtype"] = self._normalize_dtype(dtype)
            self._rm_models.append(HFRewardModel(**spec_copy))

        self._weaver_reward = None
        if self.use_weaver:
            try:
                from forge.reward.weaver_full import WeaverFullReward
            except ImportError as exc:  # pragma: no cover - import guard
                raise RuntimeError(
                    "Weaver support requires installing the scaling-verification "
                    "dependency via the 'weaver' extra: pip install -e .[weaver]"
                ) from exc
            self._weaver_reward = WeaverFullReward(**(self.weaver_cfg or {}))

    def __call__(self, prompt: str, response: str, target: Optional[str] = None) -> float:
        prompts = [prompt]
        responses = [response]
        targets = [target] if target is not None else None

        raw_signals: Dict[str, float] = {}
        per_model_scores: List[float] = []
        for name, model in zip(self._rm_names, self._rm_models):
            score = model(prompts, responses, targets)
            scalar = self._scalarize_score(score)
            raw_signals[name] = scalar
            per_model_scores.append(scalar)

        if self._weaver_reward is not None:
            scores = self._weaver_reward.score_candidates(
                prompt=prompt,
                candidates=[response],
                meta={"raw_signals": [raw_signals]},
            )
            return float(scores[0])

        stacked = torch.tensor(per_model_scores, dtype=torch.float32)
        return float(self._reduce_scores(stacked))

    @staticmethod
    def _scalarize_score(score: Any) -> float:
        if isinstance(score, torch.Tensor):
            if score.numel() == 0:
                return 0.0
            return float(score.view(-1)[0].item())
        if isinstance(score, (list, tuple)):
            if not score:
                return 0.0
            return float(score[0])
        return float(score)

    def _reduce_scores(self, values: torch.Tensor) -> torch.Tensor:
        values = values.view(-1)
        if self.ensemble_reduce == "mean":
            return values.mean()
        if self.ensemble_reduce == "median":
            return values.median()
        if self.ensemble_reduce == "max":
            return values.max()
        if self.ensemble_reduce == "vote":
            return (values > 0.0).float().mean()
        raise ValueError(f"Unknown ensemble_reduce: {self.ensemble_reduce}")

    def _normalize_dtype(self, value: Any) -> Optional[torch.dtype]:
        if value is None or isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            dtype = getattr(torch, value, None)
            if isinstance(dtype, torch.dtype):
                return dtype
            raise ValueError(f"Unknown torch dtype string: {value}")
        raise TypeError(f"torch_dtype must be torch.dtype, str, or None (got {type(value)})")

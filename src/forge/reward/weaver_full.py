from __future__ import annotations
import importlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

# If your project has a BaseReward (or similar), import it here.
# If not, this class can be returned by your reward factory directly.
try:
    from .base import BaseReward  # adjust if your base class has a different name/path
except Exception:
    class BaseReward:  # minimal shim if you don't have a base class
        def __init__(self, **kwargs): ...
        def score_candidates(self, prompt: str, candidates: List[str], **kwargs) -> List[float]:
            raise NotImplementedError
        def select_index(self, prompt: str, candidates: List[str], **kwargs) -> int:
            scores = self.score_candidates(prompt, candidates, **kwargs)
            return max(range(len(scores)), key=lambda i: scores[i])

@dataclass
class WeaverFullReward(BaseReward):
    """
    Full Weaver Selection (Stage-2) adapter.

    Expected usage:
      - You already compute per-candidate weak verifier signals (RMs, LM judges, tools).
      - Pass them in via `meta["raw_signals"]`: a list with len == len(candidates),
        each an immutable mapping {verifier_name: numeric_value}.
      - We (optionally) binarize per a threshold, and call Weaver Selection to get a score.
    """
    # binarization thresholds; e.g., {"rm:gpt-rm":0.5, "judge:llama-8b":4}
    thresholds: Mapping[str, float] = field(default_factory=dict)

    # toggles
    use_binarize: bool = True
    use_zscore: bool = False

    # how to find Weaver
    weaver_pkg: str = "scaling_verification"
    prefer_subprocess: bool = True

    # pass-thru flags to Weaver's selection runner (e.g., seed, etc.)
    extra_args: Dict[str, Any] = field(default_factory=dict)

    # optional: if your factory passes arbitrary kwargs
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.thresholds = kwargs.get("thresholds", {})
        self.use_binarize = kwargs.get("use_binarize", True)
        self.use_zscore = kwargs.get("use_zscore", False)
        self.weaver_pkg = kwargs.get("weaver_pkg", "scaling_verification")
        self.prefer_subprocess = kwargs.get("prefer_subprocess", True)
        self.extra_args = kwargs.get("extra_args", {})

        # Try to import an in-proc entrypoint if available.
        self._selection_mod = None
        if not self.prefer_subprocess:
            try:
                self._selection_mod = importlib.import_module(f"{self.weaver_pkg}.selection.run")
            except Exception:
                # fall back to subprocess if import path changes
                self.prefer_subprocess = True
                self._selection_mod = None

    # ---- public API expected by your reward call-site
    def score_candidates(self, prompt: str, candidates: List[str], **kwargs) -> List[float]:
        """
        kwargs should include:
          meta["raw_signals"]: List[Dict[str, float]]
            - one dict per candidate containing weak verifier outputs
        """
        meta = kwargs.get("meta") or {}
        raw_signals: Optional[List[Dict[str, float]]] = meta.get("raw_signals")
        if not isinstance(raw_signals, list) or len(raw_signals) != len(candidates):
            raise ValueError(
                "WeaverFullReward requires meta['raw_signals']: "
                "a list of per-candidate dicts of weak verifier scores, same length as candidates."
            )

        rows = self._prepare_rows(prompt, candidates, raw_signals)
        if self.prefer_subprocess:
            return self._run_via_subprocess(rows)
        return self._run_inproc(rows)

    def select_index(self, prompt: str, candidates: List[str], **kwargs) -> int:
        scores = self.score_candidates(prompt, candidates, **kwargs)
        return max(range(len(scores)), key=lambda i: scores[i])

    # ---- helpers
    def _prepare_rows(self, prompt: str, candidates: List[str], raw_signals: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Normalize signals into the JSONL rows consumed by Weaver Selection CLI.
        Each row corresponds to one candidate.
        """
        # stable column set for all rows
        all_names = sorted({k for d in raw_signals for k in d.keys()})
        rows: List[Dict[str, Any]] = []

        for text, sig in zip(candidates, raw_signals):
            feats = {}
            for name in all_names:
                val = float(sig.get(name, 0.0))
                if self.use_binarize and name in self.thresholds:
                    thr = float(self.thresholds[name])
                    val = 1.0 if val >= thr else 0.0
                feats[name] = val
            rows.append({
                "prompt": prompt,
                "candidate": text,
                "features": feats,
            })
        return rows

    def _run_inproc(self, rows: List[Dict[str, Any]]) -> List[float]:
        """
        Optional: use an in-memory entrypoint if Weaver exposes it.
        If not available, we transparently fall back to the subprocess path.
        """
        mod = self._selection_mod
        if mod is None or not hasattr(mod, "score_rows"):
            return self._run_via_subprocess(rows)
        # Contract: scaling_verification.selection.run.score_rows(rows, **extra_args) -> List[float]
        return list(mod.score_rows(rows, **self.extra_args))

    def _run_via_subprocess(self, rows: List[Dict[str, Any]]) -> List[float]:
        """
        Stable path: mirror Weaver's README by invoking the key script:
          python -m scaling_verification.selection.run --input ... --output ...
        and then reading the scores back.
        """
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "batch.jsonl")
            out = os.path.join(td, "scores.jsonl")

            with open(inp, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            cmd = [
                "python", "-m", f"{self.weaver_pkg}.selection.run",
                "--input", inp,
                "--output", out,
            ]
            if self.use_zscore:
                cmd += ["--zscore"]
            # pass-through any extra flags (e.g., --seed 123)
            for k, v in (self.extra_args or {}).items():
                if isinstance(v, bool):
                    if v: cmd += [f"--{k}"]
                else:
                    cmd += [f"--{k}", str(v)]

            subprocess.run(cmd, check=True)

            scores: List[float] = []
            with open(out, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    scores.append(float(obj["score"]))
            return scores

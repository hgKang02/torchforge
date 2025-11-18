"""
Lightweight Weaver-compatible utilities bundled with Forge.

We only implement the pieces that Forge relies on:

  - ``python -m weaver.selection.run`` CLI that consumes JSONL inputs
    and emits a JSONL of scores.
  - ``weaver.selection.run.score_rows`` helper so callers can import the
    scorer directly instead of invoking the subprocess.

This mirrors the public ``scaling_verification`` interface so existing
Forge configuration knobs (``weaver_pkg``) keep working even on systems
where the full Stanford implementation cannot be installed (e.g. arm64).
"""

from . import selection

__all__ = ["selection"]

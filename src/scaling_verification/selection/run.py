"""
Bridge ``python -m scaling_verification.selection.run`` to the bundled
lightweight Weaver implementation (``weaver.selection.run``).
"""

from weaver.selection.run import main, score_rows

__all__ = ["main", "score_rows"]


if __name__ == "__main__":  # pragma: no cover
    main()

from __future__ import annotations

import json
from pathlib import Path

from weaver.selection.run import main, score_rows


def _sample_rows() -> list[dict]:
    return [
        {
            "prompt": "p",
            "candidate": "a",
            "features": {"rm_a": 0.9, "rm_b": 0.4},
        },
        {
            "prompt": "p",
            "candidate": "b",
            "features": {"rm_a": 0.1, "rm_b": 0.8},
        },
    ]


def test_score_rows_prefers_higher_signal() -> None:
    rows = _sample_rows()
    scores = score_rows(rows, baseline=0.5)
    assert len(scores) == len(rows)
    # Candidate with larger combined verifier support should score higher.
    assert scores[0] > scores[1]


def test_cli_roundtrip(tmp_path: Path) -> None:
    rows = _sample_rows()
    input_path = tmp_path / "rows.jsonl"
    output_path = tmp_path / "scores.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        for obj in rows:
            handle.write(json.dumps(obj) + "\n")

    main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--baseline",
            "0.5",
            "--method",
            "auto",
        ]
    )

    with output_path.open("r", encoding="utf-8") as handle:
        scores = [json.loads(line)["score"] for line in handle]

    assert len(scores) == len(rows)
    assert scores[0] > scores[1]

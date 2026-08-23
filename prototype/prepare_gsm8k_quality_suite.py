#!/usr/bin/env python3
"""Create a deterministic dAI quality suite from the immutable GSM8K test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

GSM8K_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{GSM8K_COMMIT}/grade_school_math/data/test.jsonl"
)
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"


def read_source(source: Path | None, timeout: float) -> bytes:
    if source is not None:
        return source.read_bytes()
    with urllib.request.urlopen(GSM8K_URL, timeout=timeout) as response:
        return response.read()


def extract_expected(answer: str) -> float:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*$", answer)
    if not match:
        raise ValueError("GSM8K answer does not end in a numeric #### result")
    return float(match.group(1).replace(",", ""))


def convert(source_bytes: bytes, limit: int, seed: int) -> list[dict]:
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != GSM8K_SHA256:
        raise ValueError(f"GSM8K source hash mismatch: expected {GSM8K_SHA256}, got {digest}")
    rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line]
    if limit <= 0 or limit > len(rows):
        raise ValueError(f"limit must be between 1 and {len(rows)}")
    indices = sorted(random.Random(seed).sample(range(len(rows)), limit))
    return [{
        "id": f"gsm8k-test-{index:04d}",
        "category": "gsm8k",
        "prompt": (
            "Solve this grade-school math problem. Show concise calculations, then put "
            "the final numeric answer on the last line.\n\n" + rows[index]["question"]
        ),
        "scorer": {
            "type": "numeric",
            "expected": extract_expected(rows[index]["answer"]),
            "tolerance": 1e-9,
        },
        "max_tokens": 256,
        "source": {
            "dataset": "GSM8K",
            "split": "test",
            "row": index,
            "commit": GSM8K_COMMIT,
        },
    } for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Use an already-downloaded official test.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cases = convert(
        read_source(Path(args.source) if args.source else None, args.timeout),
        args.limit,
        args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(case, separators=(",", ":")) + "\n" for case in cases),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "cases": len(cases),
        "source_url": GSM8K_URL,
        "source_sha256": GSM8K_SHA256,
        "suite_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

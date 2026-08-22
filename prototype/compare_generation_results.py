#!/usr/bin/env python3
"""Combine same-harness generation results and compute median speedups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("results", nargs="+")
    args = parser.parse_args()

    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.results]
    baseline = reports[0]
    if baseline["variant"] != "baseline":
        raise ValueError("the first report must be the baseline")
    invariant_fields = ("schema", "prompt_tokens", "prompt_sha256", "prompt_source", "max_tokens", "repetitions")
    for report in reports[1:]:
        for field in invariant_fields:
            if report.get(field) != baseline.get(field):
                raise ValueError(f"variant {report['variant']} changed {field}")

    baseline_hashes = {
        run["output_token_sha256"] for run in baseline["runs"] if run["measured"]
    }
    if len(baseline_hashes) != 1:
        raise ValueError("baseline output is not deterministic")
    baseline_tps = baseline["summary"]["output_tps"]["p50"]
    baseline_total = baseline["summary"]["total_seconds"]["p50"]
    variants = []
    for report in reports:
        output_hashes = {
            run["output_token_sha256"] for run in report["runs"] if run["measured"]
        }
        output_equivalent = output_hashes == baseline_hashes
        variants.append({
            "variant": report["variant"],
            "summary": report["summary"],
            "correctness": {
                "measured_output_hashes": sorted(output_hashes),
                "deterministic_within_variant": len(output_hashes) == 1,
                "output_equivalent_to_baseline": output_equivalent,
                "speedup_qualified": output_equivalent,
            },
            "speedup": {
                "decode_vs_baseline": report["summary"]["output_tps"]["p50"] / baseline_tps,
                "total_vs_baseline": baseline_total / report["summary"]["total_seconds"]["p50"],
            },
        })
    comparison = {
        "schema": "dai-generation-comparison.v2",
        "prompt_tokens": baseline["prompt_tokens"],
        "prompt_sha256": baseline["prompt_sha256"],
        "prompt_source": baseline.get("prompt_source", "synthetic-repeat-legacy"),
        "max_tokens": baseline["max_tokens"],
        "repetitions": baseline["repetitions"],
        "variants": variants,
    }
    Path(args.output).write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

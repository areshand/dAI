#!/usr/bin/env python3
"""Compare an optimized expert placement with bracketing trivial-placement runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def measured_hashes(report: dict) -> list[str]:
    return sorted({
        row["output_token_sha256"]
        for row in report["runs"]
        if row.get("measured")
    })


def validate_workload(reference: dict, candidate: dict) -> None:
    fields = (
        "prompt_sha256", "prompt_tokens", "max_tokens", "repetitions",
        "cache_policy", "sampling",
    )
    for field in fields:
        if candidate.get(field) != reference.get(field):
            raise ValueError(f"workload mismatch for {field}")
    if reference["prompt_tokens"] != 1000 or reference["max_tokens"] != 256:
        raise ValueError("comparison requires the 1,000-input/256-output contract")
    for report in (reference, candidate):
        measured = [row for row in report["runs"] if row.get("measured")]
        if len(measured) != 10 or any(
            row["output_tokens"] != 256
            or len(row["output_token_ids"]) != row["output_tokens"]
            or len(set(row["output_token_ids"])) <= 1
            for row in measured
        ):
            raise ValueError("comparison requires ten complete measured requests")


def run_summary(report: dict) -> dict:
    summary = report["summary"]
    return {
        "pooled_output_tps": summary["pooled_output_tps"],
        "mean_ttft_ms": summary["ttft_seconds"]["mean"] * 1000,
        "mean_total_ms": summary["total_seconds"]["mean"] * 1000,
        "unique_output_hashes": summary["unique_output_hashes"],
        "measured_output_hashes": measured_hashes(report),
    }


def compare(pre: dict, optimized: dict, post: dict, optimization: dict) -> dict:
    validate_workload(pre, optimized)
    validate_workload(pre, post)
    pre_summary = run_summary(pre)
    optimized_summary = run_summary(optimized)
    post_summary = run_summary(post)
    bracket_tps = (pre_summary["pooled_output_tps"] + post_summary["pooled_output_tps"]) / 2
    baseline_drift = post_summary["pooled_output_tps"] / pre_summary["pooled_output_tps"] - 1
    return {
        "schema": "dai-expert-placement-comparison.v1",
        "workload": {
            "input_tokens": pre["prompt_tokens"],
            "output_tokens": pre["max_tokens"],
            "measured_requests_per_cell": pre["repetitions"],
            "cache_policy": pre["cache_policy"],
            "batch": 1,
        },
        "execution_order": ["trivial-pre", "optimized", "trivial-post"],
        "trivial_pre": pre_summary,
        "optimized": optimized_summary,
        "trivial_post": post_summary,
        "bracketed_trivial_pooled_output_tps": bracket_tps,
        "baseline_drift_fraction": baseline_drift,
        "baseline_drift_within_5_percent": abs(baseline_drift) <= 0.05,
        "optimized_speedup_vs_bracketed_trivial": (
            optimized_summary["pooled_output_tps"] / bracket_tps
        ),
        "optimized_ttft_ratio_vs_bracketed_trivial": (
            optimized_summary["mean_ttft_ms"]
            / ((pre_summary["mean_ttft_ms"] + post_summary["mean_ttft_ms"]) / 2)
        ),
        "output_hash_sets_match": (
            pre_summary["measured_output_hashes"]
            == optimized_summary["measured_output_hashes"]
            == post_summary["measured_output_hashes"]
        ),
        "predicted_routing": {
            "baseline_held_out": optimization["baseline"]["held_out"],
            "optimized_held_out": optimization["optimized"]["held_out"],
        },
        "interpretation_boundary": optimization["runtime_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trivial-pre", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--trivial-post", type=Path, required=True)
    parser.add_argument("--optimization-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        read_json(args.trivial_pre),
        read_json(args.optimized),
        read_json(args.trivial_post),
        read_json(args.optimization_report),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

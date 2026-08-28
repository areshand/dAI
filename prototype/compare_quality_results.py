#!/usr/bin/env python3
"""Apply paired quality non-inferiority and interactive-speed gates."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from generation_benchmark import percentile


def paired_bootstrap_delta(
    baseline: list[float], candidate: list[float], samples: int, seed: int
) -> dict[str, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired bootstrap needs equal non-empty inputs")
    differences = [right - left for left, right in zip(baseline, candidate)]
    rng = random.Random(seed)
    bootstrapped = [
        statistics.fmean(rng.choice(differences) for _ in differences)
        for _ in range(samples)
    ]
    return {
        "estimate": statistics.fmean(differences),
        "ci95_low": percentile(bootstrapped, 0.025),
        "ci95_high": percentile(bootstrapped, 0.975),
    }


def analyze(
    quality_reports: list[dict],
    generation_reports: list[dict],
    margin: float,
    min_cases: int,
    target_tps: float,
    max_event_gap_ms: float,
    max_ttft_ms: float,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    if not quality_reports or quality_reports[0]["variant"] != "baseline":
        raise ValueError("the first quality report must be the baseline")
    generation_by_variant = {report["variant"]: report for report in generation_reports}
    if len(generation_by_variant) != len(generation_reports):
        raise ValueError("duplicate generation variant")

    baseline = quality_reports[0]
    baseline_cases = {case["id"]: case for case in baseline["cases"]}
    case_ids = sorted(baseline_cases)
    baseline_scores = [float(baseline_cases[case_id]["score"]) for case_id in case_ids]
    cells = []
    for index, report in enumerate(quality_reports):
        if report["dataset_sha256"] != baseline["dataset_sha256"]:
            raise ValueError(f"variant {report['variant']} changed the quality dataset")
        cases = {case["id"]: case for case in report["cases"]}
        if sorted(cases) != case_ids:
            raise ValueError(f"variant {report['variant']} changed the quality case ids")
        candidate_scores = [float(cases[case_id]["score"]) for case_id in case_ids]
        delta = paired_bootstrap_delta(
            baseline_scores, candidate_scores, bootstrap_samples, seed + index
        )
        sample_sufficient = len(case_ids) >= min_cases
        noninferior = sample_sufficient and delta["ci95_low"] >= -margin

        generation = generation_by_variant.get(report["variant"])
        if generation is None:
            raise ValueError(f"missing generation report for {report['variant']}")
        mean_tps = float(generation["summary"]["output_tps"]["mean"])
        event_gap = generation["summary"]["stream_event_interarrival_seconds"]
        ttft = generation["summary"]["ttft_seconds"]
        event_gap_p99_ms = float(event_gap["p99"]) * 1000.0
        ttft_p99_ms = float(ttft["p99"]) * 1000.0
        tps_pass = mean_tps >= target_tps
        stream_pause_pass = event_gap_p99_ms <= max_event_gap_ms
        ttft_pass = ttft_p99_ms <= max_ttft_ms

        regressions = sum(
            right < left for left, right in zip(baseline_scores, candidate_scores)
        )
        gains = sum(right > left for left, right in zip(baseline_scores, candidate_scores))
        cells.append({
            "variant": report["variant"],
            "cases": len(case_ids),
            "quality": {
                "baseline_mean_score": statistics.fmean(baseline_scores),
                "candidate_mean_score": statistics.fmean(candidate_scores),
                "paired_delta": delta,
                "regressions": regressions,
                "gains": gains,
            },
            "performance": {
                "mean_output_tps": mean_tps,
                "event_gap_p99_ms": event_gap_p99_ms,
                "ttft_p99_ms": ttft_p99_ms,
            },
            "gates": {
                "sample_sufficient": sample_sufficient,
                "quality_noninferior": noninferior,
                "mean_output_tps": tps_pass,
                "stream_pause_sla": stream_pause_pass,
                "ttft_sla": ttft_pass,
                "quality_qualified_100_tps": (
                    noninferior and tps_pass and stream_pause_pass and ttft_pass
                ),
            },
        })
    return {
        "schema": "dai-quality-speed-gates.v1",
        "dataset_name": baseline["dataset_name"],
        "dataset_sha256": baseline["dataset_sha256"],
        "quality_noninferiority_margin": margin,
        "minimum_cases": min_cases,
        "target_mean_output_tps": target_tps,
        "max_event_gap_p99_ms": max_event_gap_ms,
        "max_ttft_p99_ms": max_ttft_ms,
        "bootstrap_samples": bootstrap_samples,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-report", action="append", required=True)
    parser.add_argument("--generation-report", action="append", required=True)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--min-cases", type=int, default=100)
    parser.add_argument("--target-tps", type=float, default=100.0)
    parser.add_argument("--max-event-gap-ms", type=float, default=100.0)
    parser.add_argument("--max-ttft-ms", type=float, default=250.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.margin < 1 or args.min_cases <= 0 or args.bootstrap_samples <= 0:
        parser.error("margin, min-cases, and bootstrap-samples are invalid")
    quality_reports = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in args.quality_report
    ]
    generation_reports = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in args.generation_report
    ]
    result = analyze(
        quality_reports,
        generation_reports,
        args.margin,
        args.min_cases,
        args.target_tps,
        args.max_event_gap_ms,
        args.max_ttft_ms,
        args.bootstrap_samples,
        args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Apply correctness, latency, SLA, and machine-causality gates to v2 runs."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from generation_benchmark import distribution, percentile


def measured_runs(report: dict) -> list[dict]:
    return [run for run in report["runs"] if run["measured"]]


def token_period_ms(run: dict) -> float:
    output_tokens = int(run["output_tokens"])
    if output_tokens < 2:
        raise ValueError("a measured run needs at least two output tokens")
    return 1000.0 * float(run["stream_seconds"]) / (output_tokens - 1)


def bootstrap_mean_difference(
    baseline: list[float], candidate: list[float], samples: int, seed: int
) -> dict[str, float]:
    if not baseline or not candidate:
        raise ValueError("bootstrap inputs must be non-empty")
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        baseline_sample = [rng.choice(baseline) for _ in baseline]
        candidate_sample = [rng.choice(candidate) for _ in candidate]
        differences.append(
            statistics.fmean(candidate_sample) - statistics.fmean(baseline_sample)
        )
    return {
        "estimate": statistics.fmean(candidate) - statistics.fmean(baseline),
        "ci95_low": percentile(differences, 0.025),
        "ci95_high": percentile(differences, 0.975),
    }


def parse_machine_counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        variant, separator, count = value.partition("=")
        if not separator or not variant:
            raise ValueError(f"invalid --machine-count value: {value!r}")
        result[variant] = int(count)
    return result


def analyze(
    reports: list[dict],
    machine_counts: dict[str, int],
    minimum_improvement_fraction: float,
    equivalence_fraction: float,
    sla_ms: float,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    if not reports or reports[0]["variant"] != "baseline":
        raise ValueError("the first report must be the baseline")
    baseline = reports[0]
    baseline_runs = measured_runs(baseline)
    baseline_periods = [token_period_ms(run) for run in baseline_runs]
    baseline_mean = statistics.fmean(baseline_periods)
    delta_ms = baseline_mean * minimum_improvement_fraction
    epsilon_ms = baseline_mean * equivalence_fraction
    baseline_hashes = {run["output_token_sha256"] for run in baseline_runs}
    if len(baseline_hashes) != 1:
        raise ValueError("baseline output is not deterministic")

    cells = []
    for index, report in enumerate(reports):
        runs = measured_runs(report)
        periods = [token_period_ms(run) for run in runs]
        output_hashes = {run["output_token_sha256"] for run in runs}
        valid_trace_runs = [
            run for run in runs if run["stream_trace"]["client_token_itl_valid"]
        ]
        itls_ms = [
            1000.0 * value
            for run in valid_trace_runs
            for value in run["stream_trace"]["client_token_itl_seconds"]
        ]
        latency_delta = bootstrap_mean_difference(
            baseline_periods, periods, bootstrap_samples, seed + index
        )
        improvement = latency_delta["ci95_high"] <= -delta_ms
        equivalence = (
            latency_delta["ci95_low"] >= -epsilon_ms
            and latency_delta["ci95_high"] <= epsilon_ms
        )
        exact = output_hashes == baseline_hashes
        all_traces_valid = len(valid_trace_runs) == len(runs)
        itl_distribution = distribution(itls_ms)
        sla_pass = bool(itls_ms) and all_traces_valid and (
            float(itl_distribution["p95"]) < sla_ms
            and float(itl_distribution["p99"]) < sla_ms
        )
        added_machines = machine_counts.get(report["variant"], 0)
        cells.append({
            "variant": report["variant"],
            "added_machines": added_machines,
            "machine_causal_claim_eligible": added_machines > 0,
            "correctness": {
                "deterministic": len(output_hashes) == 1,
                "exact_output_equal_to_baseline": exact,
                "hashes": sorted(output_hashes),
            },
            "mean_token_period_ms": distribution(periods),
            "latency_delta_ms": latency_delta,
            "client_token_itl_ms": itl_distribution,
            "valid_token_trace_runs": len(valid_trace_runs),
            "measured_runs": len(runs),
            "deadline_miss_fraction": (
                sum(value >= sla_ms for value in itls_ms) / len(itls_ms)
                if itls_ms else None
            ),
            "gates": {
                "minimum_improvement": improvement,
                "equivalent_to_baseline": equivalence,
                "client_itl_sla": sla_pass,
                "correctness_qualified_improvement": exact and improvement and sla_pass,
                "machine_causal_improvement": (
                    added_machines > 0 and exact and improvement and sla_pass
                ),
            },
        })
    return {
        "schema": "dai-research-gate-analysis.v1",
        "baseline_mean_token_period_ms": baseline_mean,
        "minimum_worthwhile_improvement_ms": delta_ms,
        "equivalence_half_width_ms": epsilon_ms,
        "client_itl_sla_ms": sla_ms,
        "bootstrap_samples": bootstrap_samples,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--machine-count", action="append", default=[], metavar="VARIANT=N")
    parser.add_argument("--minimum-improvement-fraction", type=float, default=0.10)
    parser.add_argument("--equivalence-fraction", type=float, default=0.05)
    parser.add_argument("--sla-ms", type=float, default=100.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.reports]
    result = analyze(
        reports,
        parse_machine_counts(args.machine_count),
        args.minimum_improvement_fraction,
        args.equivalence_fraction,
        args.sla_ms,
        args.bootstrap_samples,
        args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

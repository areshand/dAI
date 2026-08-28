#!/usr/bin/env python3
"""Compare the bracketed stock and sparse EP-combine experiment cells."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from statistics import mean


METRICS_PATTERN = re.compile(r"dai_sparse_ep_metrics=(\{.*\})")


def network_bytes(path: Path) -> int:
    interfaces = json.loads(path.read_text(encoding="utf-8"))
    return sum(
        int(item.get("stats64", {}).get("rx", {}).get("bytes", 0))
        + int(item.get("stats64", {}).get("tx", {}).get("bytes", 0))
        for item in interfaces
        if item.get("ifname") != "lo"
    )


def load_benchmark(path: Path) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    measured = [run for run in result["runs"] if run["measured"]]
    if (
        result["prompt_tokens"] != 1000
        or result["max_tokens"] != 256
        or result["repetitions"] != 10
        or result["cache_policy"] != "cold"
        or len(measured) != 10
        or any(
            run["output_tokens"] != 256
            or len(run["output_token_ids"]) != 256
            or len(set(run["output_token_ids"])) <= 1
            for run in measured
        )
    ):
        raise ValueError(f"invalid benchmark contract: {path}")
    summary = result["summary"]
    return {
        "path": str(path),
        "pooled_output_tps": summary["pooled_output_tps"],
        "mean_ttft_ms": summary["ttft_seconds"]["mean"] * 1000,
        "mean_total_ms": summary["total_seconds"]["mean"] * 1000,
        "p95_itl_ms": summary["client_token_itl_seconds"]["p95"] * 1000,
        "unique_output_hashes": summary["unique_output_hashes"],
        "output_hashes": sorted({run["output_token_sha256"] for run in measured}),
    }


def bracket(left: dict, right: dict) -> dict:
    return {
        key: mean([left[key], right[key]])
        for key in ("pooled_output_tps", "mean_ttft_ms", "mean_total_ms", "p95_itl_ms")
    }


def phase_network_bytes(artifact_dir: Path, phase: str, workers: int = 4) -> int:
    total = 0
    for rank in range(workers):
        if phase == "full-trivial-pre":
            before_name = f"rank-{rank}-before-network.json"
            after_name = f"rank-{rank}-after-network.json"
        else:
            before_name = f"rank-{rank}-{phase}-before-network.json"
            after_name = f"rank-{rank}-{phase}-after-network.json"
        before = network_bytes(artifact_dir / before_name)
        after = network_bytes(artifact_dir / after_name)
        if after < before:
            raise ValueError(f"network counter decreased for {phase} rank {rank}")
        total += after - before
    return total


def last_sparse_metrics(path: Path) -> dict | None:
    last = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = METRICS_PATTERN.search(line)
        if match:
            last = ast.literal_eval(match.group(1))
    return last


def percent_change(value: float, baseline: float) -> float:
    return (value / baseline - 1) * 100


def analyze(
    artifact_dir: Path,
    full_pre_path: Path,
    sparse_pre_path: Path,
    sparse_optimized_path: Path,
    sparse_post_path: Path,
    full_post_path: Path,
) -> dict:
    cells = {
        "full_trivial_pre": load_benchmark(full_pre_path),
        "sparse_trivial_pre": load_benchmark(sparse_pre_path),
        "sparse_optimized": load_benchmark(sparse_optimized_path),
        "sparse_trivial_post": load_benchmark(sparse_post_path),
        "full_trivial_post": load_benchmark(full_post_path),
    }
    full_reference = bracket(cells["full_trivial_pre"], cells["full_trivial_post"])
    sparse_reference = bracket(
        cells["sparse_trivial_pre"], cells["sparse_trivial_post"]
    )
    for phase, key in (
        ("full-trivial-pre", "full_trivial_pre"),
        ("sparse-trivial-pre", "sparse_trivial_pre"),
        ("sparse-optimized", "sparse_optimized"),
        ("sparse-trivial-post", "sparse_trivial_post"),
        ("full-trivial-post", "full_trivial_post"),
    ):
        cells[key]["network_bytes"] = phase_network_bytes(artifact_dir, phase)

    metrics = {}
    for phase in ("sparse-trivial-pre", "sparse-optimized", "sparse-trivial-post"):
        phase_metrics = []
        for rank in range(4):
            value = last_sparse_metrics(
                artifact_dir / f"rank-{rank}-{phase}-after-server.log"
            )
            if value is not None:
                phase_metrics.append(value)
        metrics[phase.replace("-", "_")] = phase_metrics

    optimized = cells["sparse_optimized"]
    sparse_trivial_network = mean(
        [
            cells["sparse_trivial_pre"]["network_bytes"],
            cells["sparse_trivial_post"]["network_bytes"],
        ]
    )
    return {
        "schema": "dai-sparse-ep-combine-comparison.v1",
        "contract": {
            "model": "Qwen3-30B-A3B",
            "workers": 4,
            "gpu": "NVIDIA L4 24GB-class",
            "batch": 1,
            "input_tokens": 1000,
            "output_tokens": 256,
            "cache_policy": "cold",
            "scope": "single-token decode output combine; stock full-rank input gather retained",
        },
        "cells": cells,
        "bracketed_references": {
            "full_trivial": full_reference,
            "sparse_trivial": sparse_reference,
        },
        "network_evidence": {
            "sparse_trivial_bracketed_bytes": sparse_trivial_network,
            "sparse_optimized_bytes": optimized["network_bytes"],
            "same_scope_full_trivial_post_bytes": cells["full_trivial_post"][
                "network_bytes"
            ],
            "sparse_optimized_vs_sparse_trivial_percent": percent_change(
                optimized["network_bytes"], sparse_trivial_network
            ),
            "sparse_trivial_vs_same_scope_full_post_percent": percent_change(
                sparse_trivial_network,
                cells["full_trivial_post"]["network_bytes"],
            ),
            "excluded_from_protocol_network_comparison": {
                "cell": "full_trivial_pre",
                "reason": "its counter interval includes the initial 32-token stock smoke",
            },
        },
        "effects_percent": {
            "sparse_protocol_vs_full_trivial_tps": percent_change(
                sparse_reference["pooled_output_tps"], full_reference["pooled_output_tps"]
            ),
            "optimized_placement_with_sparse_protocol_tps": percent_change(
                optimized["pooled_output_tps"], sparse_reference["pooled_output_tps"]
            ),
            "optimized_placement_with_sparse_protocol_total_latency": percent_change(
                optimized["mean_total_ms"], sparse_reference["mean_total_ms"]
            ),
            "optimized_placement_with_sparse_protocol_network": percent_change(
                optimized["network_bytes"],
                sparse_trivial_network,
            ),
            "full_reference_drift_tps": percent_change(
                cells["full_trivial_post"]["pooled_output_tps"],
                cells["full_trivial_pre"]["pooled_output_tps"],
            ),
            "sparse_reference_drift_tps": percent_change(
                cells["sparse_trivial_post"]["pooled_output_tps"],
                cells["sparse_trivial_pre"]["pooled_output_tps"],
            ),
        },
        "sparse_metrics_snapshots": metrics,
        "correctness_boundary": {
            "structural_gate": "all measured requests returned 256 non-collapsed server token IDs",
            "exact_token_hash_equivalence": len(
                set(cells["full_trivial_pre"]["output_hashes"])
                & set(cells["sparse_trivial_pre"]["output_hashes"])
            )
            > 0,
            "note": "Different BF16 reduction orders can change token IDs; this performance run is not a full agentic-quality noninferiority evaluation.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--full-pre", type=Path, required=True)
    parser.add_argument("--sparse-pre", type=Path, required=True)
    parser.add_argument("--sparse-optimized", type=Path, required=True)
    parser.add_argument("--sparse-post", type=Path, required=True)
    parser.add_argument("--full-post", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.artifact_dir,
        args.full_pre,
        args.sparse_pre,
        args.sparse_optimized,
        args.sparse_post,
        args.full_post,
    )
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

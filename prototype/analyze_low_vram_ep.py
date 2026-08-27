#!/usr/bin/env python3
"""Analyze a low-VRAM EP run and its rank-level participation evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def network_bytes(path: Path) -> int:
    total = 0
    for interface in read_json(path):
        if interface.get("ifname") == "lo":
            continue
        stats = interface.get("stats64") or interface.get("stats") or {}
        total += int((stats.get("rx") or {}).get("bytes", 0))
        total += int((stats.get("tx") or {}).get("bytes", 0))
    return total


def gpu_memory_mib(path: Path) -> int:
    row = next(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    return int(row[2].strip().removesuffix(" MiB"))


def measured_hashes(report: dict) -> list[str]:
    return sorted({
        run["output_token_sha256"]
        for run in report["runs"]
        if run.get("measured")
    })


def summary(report: dict) -> dict:
    values = report["summary"]
    return {
        "pooled_output_tps": values["pooled_output_tps"],
        "mean_ttft_ms": values["ttft_seconds"]["mean"] * 1000,
        "p99_ttft_ms": values["ttft_seconds"]["p99"] * 1000,
        "mean_total_ms": values["total_seconds"]["mean"] * 1000,
        "p99_total_ms": values["total_seconds"]["p99"] * 1000,
        "unique_output_hashes": values["unique_output_hashes"],
    }


def analyze(manifest: dict, distributed: dict, baseline: dict, artifact_dir: Path) -> dict:
    workers = int(manifest["worker_count"])
    network_deltas = []
    peak_memory = []
    for rank in range(workers):
        before = network_bytes(artifact_dir / f"rank-{rank}-before-network.json")
        after = network_bytes(artifact_dir / f"rank-{rank}-after-network.json")
        network_deltas.append({
            "rank": rank,
            "bytes": max(0, after - before),
        })
        peak_memory.append({
            "rank": rank,
            "mib": max(
                gpu_memory_mib(artifact_dir / f"rank-{rank}-before-gpu.csv"),
                gpu_memory_mib(artifact_dir / f"rank-{rank}-after-gpu.csv"),
            ),
        })

    distributed_summary = summary(distributed)
    baseline_summary = summary(baseline)
    prompt_match = distributed["prompt_sha256"] == baseline["prompt_sha256"]
    output_hash_match = measured_hashes(distributed) == measured_hashes(baseline)
    # One MiB is far above SSM heartbeat noise during this short window and
    # deliberately conservative relative to EP collective traffic.
    all_network_active = all(item["bytes"] > 1024 * 1024 for item in network_deltas)
    memory_contract = all(
        item["mib"] <= manifest.get(
            "gpu_vram_limit_mib_per_worker", manifest["gpu_vram_mib_per_worker"]
        )
        for item in peak_memory
    )
    parallelism = manifest["parallelism"]
    joint = (
        workers == 4
        and parallelism == {"tp": 4, "dp": 4, "ep": 4, "attention_tp": 1}
        and all_network_active
        and memory_contract
        and distributed["prompt_tokens"] == 1000
        and distributed["max_tokens"] == 256
    )
    return {
        "schema": "dai-low-vram-ep-analysis.v1",
        "joint_inference_proven": joint,
        "proof_contract": {
            "no_rank_can_hold_full_bf16_checkpoint": manifest["gpu_vram_mib_per_worker"] < 57 * 1024,
            "worker_count": workers,
            "gpu_vram_mib_per_worker": manifest["gpu_vram_mib_per_worker"],
            "parallelism": parallelism,
            "all_ranks_network_active": all_network_active,
            "gpu_memory_within_contract": memory_contract,
        },
        "all_ranks_network_active": all_network_active,
        "network_bytes_during_evaluation": network_deltas,
        "peak_gpu_memory_mib": peak_memory,
        "prompt_hash_match": prompt_match,
        "output_token_hash_match": output_hash_match,
        "distributed": distributed_summary,
        "baseline": baseline_summary,
        "speed_ratio": (
            distributed_summary["pooled_output_tps"]
            / baseline_summary["pooled_output_tps"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--distributed", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        read_json(args.manifest),
        read_json(args.distributed),
        read_json(args.baseline),
        args.artifact_dir,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a capacity-constrained SGLang expert placement from routed-expert traces."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


def read_route_splits(path: Path, num_layers: int, train_requests: int):
    request_split: dict[int, str] = {}
    request_ids = {"train": [], "test": []}
    edges = {
        "train": [[] for _ in range(num_layers)],
        "test": [[] for _ in range(num_layers)],
    }
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not row.get("measured"):
                continue
            request_index = int(row["request_index"])
            if request_index not in request_split:
                split = "train" if len(request_ids["train"]) < train_requests else "test"
                request_split[request_index] = split
                request_ids[split].append(request_index)
            split = request_split[request_index]
            layer = int(row["layer"])
            if not 0 <= layer < num_layers:
                raise ValueError(f"layer {layer} is outside [0, {num_layers})")
            experts = tuple(dict.fromkeys(int(value) for value in row["expert_ids"]))
            edges[split][layer].append(experts)
    if not request_ids["train"] or not request_ids["test"]:
        raise ValueError("the trace must contain both training and held-out measured requests")
    return edges, request_ids


def activation_counts(layer_edges: list[tuple[int, ...]], experts: int) -> list[int]:
    counts = [0] * experts
    for edge in layer_edges:
        for expert in edge:
            if not 0 <= expert < experts:
                raise ValueError(f"expert {expert} is outside [0, {experts})")
            counts[expert] += 1
    return counts


def pair_affinity(layer_edges: list[tuple[int, ...]], experts: int) -> list[list[int]]:
    affinity = [[0] * experts for _ in range(experts)]
    for edge in layer_edges:
        for left, right in combinations(edge, 2):
            affinity[left][right] += 1
            affinity[right][left] += 1
    return affinity


def initial_balanced_assignment(counts: list[int], workers: int) -> list[int]:
    experts = len(counts)
    if experts % workers:
        raise ValueError("experts must divide evenly across workers")
    capacity = experts // workers
    assignment = [-1] * experts
    loads = [0] * workers
    slots = [0] * workers
    for expert in sorted(range(experts), key=lambda item: (-counts[item], item)):
        worker = min(
            (rank for rank in range(workers) if slots[rank] < capacity),
            key=lambda rank: (loads[rank], slots[rank], rank),
        )
        assignment[expert] = worker
        loads[worker] += counts[expert]
        slots[worker] += 1
    return assignment


def worker_loads(assignment: list[int], counts: list[int], workers: int) -> list[int]:
    loads = [0] * workers
    for expert, worker in enumerate(assignment):
        loads[worker] += counts[expert]
    return loads


def pair_cut(assignment: list[int], affinity: list[list[int]]) -> int:
    return sum(
        affinity[left][right]
        for left in range(len(assignment))
        for right in range(left + 1, len(assignment))
        if assignment[left] != assignment[right]
    )


def optimize_layer(
    layer_edges: list[tuple[int, ...]],
    experts: int,
    workers: int,
    max_load_ratio: float,
) -> tuple[list[int], dict]:
    counts = activation_counts(layer_edges, experts)
    affinity = pair_affinity(layer_edges, experts)
    assignment = initial_balanced_assignment(counts, workers)
    loads = worker_loads(assignment, counts, workers)
    mean_load = sum(loads) / workers
    load_ceiling = max(loads) if mean_load == 0 else max(max(loads), mean_load * max_load_ratio)
    before_cut = pair_cut(assignment, affinity)
    swaps = 0

    # Capacity-preserving pair swaps improve co-activation locality. A hard load
    # ceiling prevents the locality objective from creating a hot-rank straggler.
    while True:
        best = None
        for left in range(experts):
            left_worker = assignment[left]
            for right in range(left + 1, experts):
                right_worker = assignment[right]
                if left_worker == right_worker:
                    continue
                new_left_load = loads[left_worker] - counts[left] + counts[right]
                new_right_load = loads[right_worker] - counts[right] + counts[left]
                if new_left_load > load_ceiling or new_right_load > load_ceiling:
                    continue
                delta = 0
                for other in range(experts):
                    if other == left or other == right:
                        continue
                    other_worker = assignment[other]
                    delta += affinity[left][other] * (
                        int(right_worker != other_worker) - int(left_worker != other_worker)
                    )
                    delta += affinity[right][other] * (
                        int(left_worker != other_worker) - int(right_worker != other_worker)
                    )
                candidate = (delta, max(new_left_load, new_right_load), left, right)
                if delta < 0 and (best is None or candidate < best):
                    best = candidate
        if best is None:
            break
        _, _, left, right = best
        left_worker = assignment[left]
        right_worker = assignment[right]
        loads[left_worker] = loads[left_worker] - counts[left] + counts[right]
        loads[right_worker] = loads[right_worker] - counts[right] + counts[left]
        assignment[left], assignment[right] = right_worker, left_worker
        swaps += 1

    return assignment, {
        "training_activations_by_worker": loads,
        "training_max_to_mean_load_ratio": max(loads) / mean_load if mean_load else 1.0,
        "pair_cut_before_locality_swaps": before_cut,
        "pair_cut_after_locality_swaps": pair_cut(assignment, affinity),
        "locality_swaps": swaps,
    }


def placement_metrics(
    split_edges: list[list[tuple[int, ...]]],
    assignments: list[list[int]],
    workers: int,
) -> dict:
    fanout = Counter()
    cross_pairs = 0
    total_pairs = 0
    max_to_mean = []
    total_worker_loads = [0] * workers
    for layer, layer_edges in enumerate(split_edges):
        loads = [0] * workers
        assignment = assignments[layer]
        for edge in layer_edges:
            edge_workers = {assignment[expert] for expert in edge}
            fanout[len(edge_workers)] += 1
            for expert in edge:
                loads[assignment[expert]] += 1
            for left, right in combinations(edge, 2):
                total_pairs += 1
                cross_pairs += assignment[left] != assignment[right]
        for worker in range(workers):
            total_worker_loads[worker] += loads[worker]
        mean = sum(loads) / workers
        max_to_mean.append(max(loads) / mean if mean else 1.0)
    rows = sum(fanout.values())
    return {
        "token_layer_rows": rows,
        "worker_fanout_token_layer_counts": {
            str(worker_count): fanout.get(worker_count, 0)
            for worker_count in range(1, workers + 1)
        },
        "mean_workers_per_token_layer": (
            sum(worker_count * count for worker_count, count in fanout.items()) / rows
            if rows else 0
        ),
        "all_workers_token_layer_fraction": fanout.get(workers, 0) / rows if rows else 0,
        "cross_worker_expert_pair_fraction": cross_pairs / total_pairs if total_pairs else 0,
        "activation_counts_by_worker": total_worker_loads,
        "mean_layer_max_to_mean_activation_ratio": sum(max_to_mean) / len(max_to_mean),
        "worst_layer_max_to_mean_activation_ratio": max(max_to_mean),
    }


def physical_to_logical(assignments: list[list[int]], workers: int) -> list[list[int]]:
    mapping = []
    for assignment in assignments:
        row = []
        for worker in range(workers):
            row.extend(expert for expert, owner in enumerate(assignment) if owner == worker)
        if sorted(row) != list(range(len(assignment))):
            raise AssertionError("placement row is not a permutation of logical experts")
        mapping.append(row)
    return mapping


def optimize(
    route_path: Path,
    num_layers: int,
    experts: int,
    workers: int,
    train_requests: int,
    max_load_ratio: float,
) -> tuple[dict, dict]:
    edges, request_ids = read_route_splits(route_path, num_layers, train_requests)
    assignments = []
    layer_details = []
    for layer in range(num_layers):
        assignment, detail = optimize_layer(
            edges["train"][layer], experts, workers, max_load_ratio
        )
        assignments.append(assignment)
        layer_details.append({"layer": layer, **detail})
    baseline = [[expert // (experts // workers) for expert in range(experts)] for _ in range(num_layers)]
    config = {"physical_to_logical_map": physical_to_logical(assignments, workers)}
    report = {
        "schema": "dai-expert-placement-optimization.v1",
        "method": "capacity-constrained pair-affinity partition with activation-load ceiling",
        "source_trace": str(route_path),
        "num_layers": num_layers,
        "experts_per_layer": experts,
        "workers": workers,
        "experts_per_worker_per_layer": experts // workers,
        "training_request_indices": request_ids["train"],
        "held_out_request_indices": request_ids["test"],
        "max_training_load_ratio": max_load_ratio,
        "runtime_boundary": (
            "SGLang moe_a2a_backend=none still uses four-rank FULL collectives; "
            "placement can reduce compute imbalance but not collective participation"
        ),
        "baseline": {
            "train": placement_metrics(edges["train"], baseline, workers),
            "held_out": placement_metrics(edges["test"], baseline, workers),
        },
        "optimized": {
            "train": placement_metrics(edges["train"], assignments, workers),
            "held_out": placement_metrics(edges["test"], assignments, workers),
        },
        "layers": layer_details,
    }
    return config, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--train-requests", type=int, default=5)
    parser.add_argument("--max-load-ratio", type=float, default=1.05)
    args = parser.parse_args()
    config, report = optimize(
        args.routes,
        args.num_layers,
        args.experts,
        args.workers,
        args.train_requests,
        args.max_load_ratio,
    )
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

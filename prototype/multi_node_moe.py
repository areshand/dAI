#!/usr/bin/env python3
"""Dependency-free multi-worker synthetic MoE placement experiment.

The coordinator measures the real application path to every worker, freezes
that discovery snapshot, derives a latency-aware placement, and compares it
with preregistered near/far/random placements in counterbalanced blocks.
Synthetic expert outputs are invariant across placements.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from two_node_moe import (
    EXPERTS,
    RemoteClient,
    expert_digest,
    inventory,
    make_payload,
    make_trace,
    summary_ms,
)


POLICY_NAMES = ("hot_near", "hot_far", "latency_aware", "random")


def parse_worker(value):
    """Parse NAME=HOST:PORT without imposing DNS naming conventions."""
    try:
        name, endpoint = value.split("=", 1)
        host, raw_port = endpoint.rsplit(":", 1)
        port = int(raw_port)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("worker must be NAME=HOST:PORT")
    if not name or not host or not (1 <= port <= 65535):
        raise argparse.ArgumentTypeError("worker must be NAME=HOST:PORT")
    return name, host, port


def build_placements(worker_names, discovery, seed):
    if len(worker_names) != 2:
        raise ValueError("the first controlled multi-node cell requires exactly two workers")
    if set(worker_names) != {"near", "far"}:
        raise ValueError("workers must be named near and far for preregistered policies")

    faster = min(worker_names, key=lambda name: discovery[name]["probe_ms"]["p50"])
    slower = next(name for name in worker_names if name != faster)
    random_workers = ["near", "near", "far", "far"]
    rng = random.Random(seed)
    while True:
        rng.shuffle(random_workers)
        if random_workers not in (["near", "near", "far", "far"],
                                   ["far", "far", "near", "near"]):
            break
    return {
        "hot_near": {0: "near", 1: "near", 2: "far", 3: "far"},
        "hot_far": {0: "far", 1: "far", 2: "near", 3: "near"},
        "latency_aware": {0: faster, 1: faster, 2: slower, 3: slower},
        "random": dict(zip(EXPERTS, random_workers)),
    }


async def connect_workers(worker_specs, concurrency, connect_timeout):
    pools = {}
    inventories = {}
    for name, host, port in worker_specs:
        clients = [RemoteClient(host, port) for _ in range(concurrency)]
        for client in clients:
            deadline = time.monotonic() + connect_timeout
            while True:
                try:
                    await client.connect()
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    await asyncio.sleep(2)
        pools[name] = clients
        inventories[name] = await clients[0].request({"op": "inventory"})
    return pools, inventories


async def close_workers(pools):
    for clients in pools.values():
        for client in clients:
            await client.close()


async def probe_worker(name, clients, samples, payload_bytes):
    latencies = []
    worker_compute = []
    residual = []
    payload = make_payload(0, payload_bytes)
    for sample_id in range(samples):
        started = time.perf_counter_ns()
        response = await clients[0].request(
            {"op": "infer", "request_id": "probe-%s-%d" % (name, sample_id), "expert_id": 0},
            payload,
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        worker_ms = float(response["worker_elapsed_ms"])
        latencies.append(elapsed_ms)
        worker_compute.append(worker_ms)
        residual.append(max(0.0, elapsed_ms - worker_ms))
    return {
        "samples": samples,
        "payload_bytes": payload_bytes,
        "probe_ms": summary_ms(latencies),
        "worker_compute_ms": summary_ms(worker_compute),
        "non_worker_residual_ms": summary_ms(residual),
    }


async def execute_policy(name, mapping, trace, pools, payload_bytes, concurrency, block, order_index):
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []
    by_worker = {worker: [] for worker in pools}
    worker_compute = {worker: [] for worker in pools}
    outputs = {}

    async def one(position, request_id, expert_id):
        async with semaphore:
            payload = make_payload(request_id, payload_bytes)
            expected = expert_digest(expert_id, payload)
            worker = mapping[expert_id]
            client_pool = pools[worker]
            started = time.perf_counter_ns()
            response = await client_pool[position % len(client_pool)].request(
                {"op": "infer", "request_id": request_id, "expert_id": expert_id}, payload
            )
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            if response["digest"] != expected:
                raise AssertionError("digest mismatch for request %d" % request_id)
            outputs[request_id] = response["digest"]
            latencies.append(elapsed_ms)
            by_worker[worker].append(elapsed_ms)
            worker_compute[worker].append(float(response["worker_elapsed_ms"]))

    started = time.perf_counter()
    await asyncio.gather(*(one(i, request_id, expert_id)
                           for i, (request_id, expert_id) in enumerate(trace)))
    wall_seconds = time.perf_counter() - started
    return {
        "policy": name,
        "block": block,
        "order_index": order_index,
        "mapping": {str(expert): worker for expert, worker in mapping.items()},
        "requests": len(trace),
        "requests_by_worker": {worker: len(values) for worker, values in by_worker.items()},
        "wall_seconds": wall_seconds,
        "requests_per_second": len(trace) / wall_seconds,
        "latency_ms": summary_ms(latencies),
        "latency_by_worker_ms": {worker: summary_ms(values) for worker, values in by_worker.items()},
        "worker_compute_ms": {worker: summary_ms(values) for worker, values in worker_compute.items()},
        "outputs": outputs,
    }


def aggregate_runs(runs):
    result = {}
    for policy in POLICY_NAMES:
        selected = [run for run in runs if run["policy"] == policy]
        result[policy] = {
            "blocks": len(selected),
            "wall_ms": summary_ms([run["wall_seconds"] * 1000 for run in selected]),
            "request_p50_ms": summary_ms([run["latency_ms"]["p50"] for run in selected]),
            "request_p95_ms": summary_ms([run["latency_ms"]["p95"] for run in selected]),
            "requests_per_second": summary_ms([run["requests_per_second"] for run in selected]),
        }
    return result


async def run_experiment(args):
    worker_specs = [parse_worker(value) for value in args.worker]
    pools, inventories = await connect_workers(
        worker_specs, args.concurrency, args.connect_timeout
    )
    try:
        discovery = {}
        for name, _, _ in worker_specs:
            discovery[name] = await probe_worker(
                name, pools[name], args.probe_samples, args.payload_bytes
            )
        placements = build_placements([item[0] for item in worker_specs], discovery, args.seed)
        trace = make_trace(args.requests, args.seed)
        runs = []
        reference_outputs = None
        rng = random.Random(args.seed + 1)
        block_orders = []
        for block in range(args.blocks):
            order = list(POLICY_NAMES)
            rng.shuffle(order)
            block_orders.append(order)
            for order_index, policy in enumerate(order):
                run = await execute_policy(
                    policy, placements[policy], trace, pools, args.payload_bytes,
                    args.concurrency, block, order_index,
                )
                if reference_outputs is None:
                    reference_outputs = run["outputs"]
                elif run["outputs"] != reference_outputs:
                    raise AssertionError("placement changed synthetic expert outputs")
                run.pop("outputs")
                runs.append(run)

        report = {
            "schema": "multi-node-synthetic-moe.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "claim_scope": "synthetic multi-worker transport/placement infrastructure only",
            "coordinator": inventory(),
            "workers": inventories,
            "parameters": {
                "workers": {name: {"host": host, "port": port} for name, host, port in worker_specs},
                "requests": args.requests,
                "payload_bytes": args.payload_bytes,
                "concurrency": args.concurrency,
                "probe_samples": args.probe_samples,
                "blocks": args.blocks,
                "seed": args.seed,
            },
            "discovery": discovery,
            "placements": {name: {str(key): value for key, value in mapping.items()}
                           for name, mapping in placements.items()},
            "block_orders": block_orders,
            "correctness": {"outputs_identical_across_all_placements": True},
            "runs": runs,
            "aggregate": aggregate_runs(runs),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "output": str(output),
            "discovery": discovery,
            "placements": report["placements"],
            "correctness": report["correctness"],
            "aggregate": report["aggregate"],
        }, indent=2, sort_keys=True))
    finally:
        await close_workers(pools)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="append", required=True, metavar="NAME=HOST:PORT")
    parser.add_argument("--requests", type=int, default=400)
    parser.add_argument("--payload-bytes", type=int, default=64 * 1024)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--probe-samples", type=int, default=30)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--connect-timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--output", default="results/multi-node-latest.json")
    args = parser.parse_args()
    if args.requests <= 0 or args.payload_bytes < 0 or args.concurrency <= 0:
        parser.error("requests and concurrency must be positive; payload-bytes must be nonnegative")
    if args.probe_samples <= 0 or args.blocks <= 0 or args.connect_timeout <= 0:
        parser.error("probe-samples, blocks, and connect-timeout must be positive")
    return args


def main():
    try:
        asyncio.run(run_experiment(parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

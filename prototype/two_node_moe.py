#!/usr/bin/env python3
"""Dependency-free two-node synthetic MoE placement experiment.

This is an infrastructure probe, not a model-quality benchmark. It keeps a
deterministic route trace and expert function fixed while swapping which
experts execute locally versus on a remote worker.
"""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import socket
import statistics
import struct
import time
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_VERSION = 1
EXPERTS = (0, 1, 2, 3)
PLACEMENTS = {
    "hot_local": {0: "local", 1: "local", 2: "remote", 3: "remote"},
    "hot_remote": {0: "remote", 1: "remote", 2: "local", 3: "local"},
}


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    fraction = pos - lo
    return ordered[lo] * (1 - fraction) + ordered[hi] * fraction


def summary_ms(values):
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def expert_digest(expert_id, payload):
    digest = hashlib.sha256()
    digest.update(struct.pack("!I", expert_id))
    digest.update(payload)
    return digest.hexdigest()


def inventory():
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


async def read_message(reader):
    raw_size = await reader.readexactly(4)
    header_size = struct.unpack("!I", raw_size)[0]
    if header_size > 64 * 1024:
        raise ValueError("header is too large")
    header = json.loads((await reader.readexactly(header_size)).decode("utf-8"))
    payload_size = int(header.get("payload_size", 0))
    if payload_size < 0 or payload_size > 64 * 1024 * 1024:
        raise ValueError("payload is too large")
    payload = await reader.readexactly(payload_size)
    return header, payload


async def write_message(writer, header, payload=b""):
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    writer.write(struct.pack("!I", len(encoded)))
    writer.write(encoded)
    writer.write(payload)
    await writer.drain()


async def handle_worker(reader, writer, worker_id, compute_ms):
    peer = writer.get_extra_info("peername")
    try:
        while True:
            header, payload = await read_message(reader)
            op = header.get("op")
            if op == "inventory":
                response = {
                    "protocol_version": PROTOCOL_VERSION,
                    "worker_id": worker_id,
                    "inventory": inventory(),
                    "payload_size": 0,
                }
            elif op == "infer":
                start_ns = time.perf_counter_ns()
                if compute_ms:
                    await asyncio.sleep(compute_ms / 1000.0)
                digest = expert_digest(int(header["expert_id"]), payload)
                response = {
                    "protocol_version": PROTOCOL_VERSION,
                    "worker_id": worker_id,
                    "request_id": header["request_id"],
                    "digest": digest,
                    "worker_elapsed_ms": (time.perf_counter_ns() - start_ns) / 1e6,
                    "payload_size": 0,
                }
            else:
                raise ValueError("unknown operation: %r" % (op,))
            await write_message(writer, response)
    except (asyncio.IncompleteReadError, ConnectionResetError):
        pass
    except Exception as exc:
        error = {"error": repr(exc), "payload_size": 0}
        try:
            await write_message(writer, error)
        except Exception:
            pass
        print("worker error from %r: %r" % (peer, exc), flush=True)
    finally:
        writer.close()
        await writer.wait_closed()


async def serve(args):
    server = await asyncio.start_server(
        lambda r, w: handle_worker(r, w, args.worker_id, args.compute_ms),
        args.bind,
        args.port,
    )
    addresses = [str(sock.getsockname()) for sock in server.sockets or []]
    print(json.dumps({"status": "ready", "worker_id": args.worker_id, "listen": addresses}), flush=True)
    async with server:
        await server.serve_forever()


class RemoteClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None
        self.lock = asyncio.Lock()

    async def connect(self):
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)

    async def request(self, header, payload=b""):
        async with self.lock:
            await write_message(self.writer, {**header, "payload_size": len(payload)}, payload)
            response, _ = await read_message(self.reader)
            if "error" in response:
                raise RuntimeError(response["error"])
            return response

    async def close(self):
        if self.writer is not None:
            self.writer.close()
            await self.writer.wait_closed()


def make_trace(requests, seed):
    rng = random.Random(seed)
    trace = []
    for request_id in range(requests):
        # Experts 0 and 1 jointly receive 80% of the fixed workload.
        bucket = rng.random()
        if bucket < 0.40:
            expert_id = 0
        elif bucket < 0.80:
            expert_id = 1
        elif bucket < 0.90:
            expert_id = 2
        else:
            expert_id = 3
        trace.append((request_id, expert_id))
    return trace


def make_payload(request_id, payload_bytes):
    seed = hashlib.sha256(("activation:%d" % request_id).encode("ascii")).digest()
    return (seed * ((payload_bytes + len(seed) - 1) // len(seed)))[:payload_bytes]


async def execute_placement(name, mapping, trace, clients, payload_bytes, concurrency):
    latencies = []
    local_latencies = []
    remote_latencies = []
    outputs = {}
    semaphore = asyncio.Semaphore(concurrency)

    async def one(position, request_id, expert_id):
        async with semaphore:
            payload = make_payload(request_id, payload_bytes)
            expected = expert_digest(expert_id, payload)
            location = mapping[expert_id]
            start_ns = time.perf_counter_ns()
            if location == "local":
                observed = expert_digest(expert_id, payload)
                worker_id = "coordinator-local"
            else:
                client = clients[position % len(clients)]
                response = await client.request(
                    {"op": "infer", "request_id": request_id, "expert_id": expert_id}, payload
                )
                observed = response["digest"]
                worker_id = response["worker_id"]
            elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
            if observed != expected:
                raise AssertionError("digest mismatch for request %d" % request_id)
            latencies.append(elapsed_ms)
            (local_latencies if location == "local" else remote_latencies).append(elapsed_ms)
            outputs[request_id] = observed
            return {"request_id": request_id, "expert_id": expert_id, "location": location,
                    "worker_id": worker_id, "elapsed_ms": elapsed_ms}

    started = time.perf_counter()
    rows = await asyncio.gather(*(one(i, rid, eid) for i, (rid, eid) in enumerate(trace)))
    wall_seconds = time.perf_counter() - started
    return {
        "placement": name,
        "mapping": {str(k): v for k, v in mapping.items()},
        "requests": len(trace),
        "local_requests": len(local_latencies),
        "remote_requests": len(remote_latencies),
        "wall_seconds": wall_seconds,
        "requests_per_second": len(trace) / wall_seconds,
        "latency_ms": summary_ms(latencies),
        "local_latency_ms": summary_ms(local_latencies),
        "remote_latency_ms": summary_ms(remote_latencies),
        "outputs": outputs,
        "trace": rows,
    }


async def run_experiment(args):
    clients = [RemoteClient(args.remote_host, args.port) for _ in range(args.concurrency)]
    for client in clients:
        await client.connect()
    try:
        remote_inventory = await clients[0].request({"op": "inventory"})
        trace = make_trace(args.requests, args.seed)
        results = []
        for name in ("hot_local", "hot_remote"):
            result = await execute_placement(
                name, PLACEMENTS[name], trace, clients, args.payload_bytes, args.concurrency
            )
            results.append(result)

        outputs_match = results[0]["outputs"] == results[1]["outputs"]
        if not outputs_match:
            raise AssertionError("placement changed synthetic expert outputs")

        compact_results = []
        for result in results:
            compact = dict(result)
            compact.pop("outputs")
            if not args.include_trace:
                compact.pop("trace")
            compact_results.append(compact)

        report = {
            "schema": "two-node-synthetic-moe.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "claim_scope": "synthetic transport/placement infrastructure only; not real-model inference",
            "coordinator": inventory(),
            "remote_worker": remote_inventory,
            "parameters": {
                "remote_host": args.remote_host,
                "port": args.port,
                "requests": args.requests,
                "payload_bytes": args.payload_bytes,
                "concurrency": args.concurrency,
                "seed": args.seed,
                "hot_expert_fraction_target": 0.80,
            },
            "correctness": {"outputs_identical_across_placements": outputs_match},
            "results": compact_results,
        }
        hot_local = results[0]["latency_ms"]["p50"]
        hot_remote = results[1]["latency_ms"]["p50"]
        report["comparison"] = {
            "p50_hot_remote_minus_hot_local_ms": hot_remote - hot_local,
            "p50_hot_remote_over_hot_local": hot_remote / hot_local if hot_local else None,
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "output": str(output_path),
            "correctness": report["correctness"],
            "comparison": report["comparison"],
            "placements": [
                {"name": r["placement"], "remote_requests": r["remote_requests"],
                 "p50_ms": r["latency_ms"]["p50"], "p95_ms": r["latency_ms"]["p95"],
                 "requests_per_second": r["requests_per_second"]}
                for r in results
            ],
        }, indent=2, sort_keys=True))
    finally:
        for client in clients:
            await client.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser("serve", help="run a synthetic expert worker")
    server.add_argument("--bind", default="0.0.0.0")
    server.add_argument("--port", type=int, default=50123)
    server.add_argument("--worker-id", default=socket.gethostname())
    server.add_argument("--compute-ms", type=float, default=0.0)

    run = subparsers.add_parser("run", help="compare hot-local and hot-remote placements")
    run.add_argument("--remote-host", required=True)
    run.add_argument("--port", type=int, default=50123)
    run.add_argument("--requests", type=int, default=400)
    run.add_argument("--payload-bytes", type=int, default=64 * 1024)
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--seed", type=int, default=20260820)
    run.add_argument("--include-trace", action="store_true")
    run.add_argument("--output", default="results/two-node-latest.json")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        if args.command == "serve":
            asyncio.run(serve(args))
        else:
            asyncio.run(run_experiment(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

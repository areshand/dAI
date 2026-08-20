#!/usr/bin/env python3
"""Extract and validate one real Qwen3 MoE expert across two machines."""

import argparse
import hashlib
import json
import math
import platform
import socket
import struct
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file


MODEL_ID = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def extract(args):
    prefix = "model.layers.%d.mlp.experts.%d" % (args.layer, args.expert)
    source_keys = {
        "gate_proj.weight": prefix + ".gate_proj.weight",
        "up_proj.weight": prefix + ".up_proj.weight",
        "down_proj.weight": prefix + ".down_proj.weight",
    }
    tensors = {}
    with safe_open(args.shard, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        missing = [key for key in source_keys.values() if key not in available]
        if missing:
            raise KeyError("missing source keys: %r" % missing)
        for short_key, source_key in source_keys.items():
            tensors[short_key] = handle.get_tensor(source_key).contiguous()

    metadata = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "layer_id": str(args.layer),
        "expert_id": str(args.expert),
        "source_shard": Path(args.shard).name,
        "source_shard_sha256": sha256_file(args.shard),
        "kernel": "down_proj(silu(gate_proj(x))*up_proj(x))",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output), metadata=metadata)
    report = {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "metadata": metadata,
        "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in tensors.items()},
        "parameter_count": sum(value.numel() for value in tensors.values()),
        "bytes": output.stat().st_size,
    }
    write_json(str(output) + ".json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def make_input(args):
    # Integer-derived values are exactly reproducible before the BF16 cast.
    values = torch.arange(args.rows * args.hidden_size, dtype=torch.int64)
    values = ((values * 37 + 11) % 2001 - 1000).to(torch.float32) / 1000.0
    hidden = values.reshape(args.rows, args.hidden_size).to(torch.bfloat16).contiguous()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"hidden_states": hidden}, str(output), metadata={
        "generator": "integer-affine-v1",
        "rows": str(args.rows),
        "hidden_size": str(args.hidden_size),
        "dtype": "bfloat16",
    })
    print(json.dumps({"output": str(output), "sha256": sha256_file(output),
                      "shape": list(hidden.shape), "dtype": str(hidden.dtype)}, indent=2))


def expert_forward(hidden, weights):
    gate = F.linear(hidden, weights["gate_proj.weight"])
    up = F.linear(hidden, weights["up_proj.weight"])
    return F.linear(F.silu(gate) * up, weights["down_proj.weight"])


def recv_exact(connection, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("connection closed with %d bytes remaining" % remaining)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_packet(connection):
    header_size = struct.unpack("!I", recv_exact(connection, 4))[0]
    if header_size > 64 * 1024:
        raise ValueError("header is too large")
    header = json.loads(recv_exact(connection, header_size).decode("utf-8"))
    payload = recv_exact(connection, int(header.get("payload_bytes", 0)))
    return header, payload


def send_packet(connection, header, payload=b""):
    header = {**header, "payload_bytes": len(payload)}
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    connection.sendall(struct.pack("!I", len(encoded)) + encoded + payload)


def tensor_to_bytes(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def bf16_from_bytes(payload, rows, hidden_size):
    expected = rows * hidden_size * 2
    if len(payload) != expected:
        raise ValueError("expected %d BF16 bytes, received %d" % (expected, len(payload)))
    return torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).clone().reshape(rows, hidden_size)


def serve_expert(args):
    weights = load_file(args.expert, device="cpu")
    weights = {key: value.to(dtype=torch.bfloat16) for key, value in weights.items()}
    hidden_size = weights["gate_proj.weight"].shape[1]
    with torch.no_grad():
        for _ in range(args.warmup):
            expert_forward(torch.zeros((1, hidden_size), dtype=torch.bfloat16), weights)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.bind, args.port))
        listener.listen()
        print(json.dumps({"status": "ready", "bind": args.bind, "port": args.port,
                          "expert_sha256": sha256_file(args.expert)}), flush=True)
        while True:
            connection, address = listener.accept()
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                while True:
                    try:
                        header, payload = recv_packet(connection)
                    except EOFError:
                        break
                    if header.get("op") != "expert_forward":
                        raise ValueError("unexpected operation")
                    preprocess_started = time.perf_counter_ns()
                    hidden = bf16_from_bytes(payload, int(header["rows"]), int(header["hidden_size"]))
                    preprocess_ms = (time.perf_counter_ns() - preprocess_started) / 1e6
                    started = time.perf_counter_ns()
                    with torch.no_grad():
                        output = expert_forward(hidden, weights)
                    compute_ms = (time.perf_counter_ns() - started) / 1e6
                    serialize_started = time.perf_counter_ns()
                    output_payload = tensor_to_bytes(output)
                    serialize_ms = (time.perf_counter_ns() - serialize_started) / 1e6
                    send_packet(connection, {
                        "request_id": header["request_id"],
                        "worker_compute_ms": compute_ms,
                        "worker_preprocess_ms": preprocess_ms,
                        "worker_serialize_ms": serialize_ms,
                        "rows": output.shape[0],
                        "hidden_size": output.shape[1],
                        "dtype": "bfloat16",
                    }, output_payload)


def remote_run(args):
    hidden = load_file(args.input, device="cpu")["hidden_states"].to(torch.bfloat16).contiguous()
    payload = tensor_to_bytes(hidden)
    samples = []
    final_output = None
    with socket.create_connection((args.host, args.port), timeout=args.timeout) as connection:
        connection.settimeout(args.timeout)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for request_id in range(args.warmup + args.repeats):
            started = time.perf_counter_ns()
            send_packet(connection, {
                "op": "expert_forward",
                "request_id": request_id,
                "rows": hidden.shape[0],
                "hidden_size": hidden.shape[1],
                "dtype": "bfloat16",
            }, payload)
            response, output_bytes = recv_packet(connection)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            final_output = bf16_from_bytes(output_bytes, response["rows"], response["hidden_size"])
            if request_id >= args.warmup:
                samples.append({
                    "request_id": request_id,
                    "round_trip_ms": elapsed_ms,
                    "worker_compute_ms": response["worker_compute_ms"],
                    "worker_preprocess_ms": response.get("worker_preprocess_ms", 0.0),
                    "worker_serialize_ms": response.get("worker_serialize_ms", 0.0),
                    "unattributed_ms": elapsed_ms - response["worker_compute_ms"]
                        - response.get("worker_preprocess_ms", 0.0)
                        - response.get("worker_serialize_ms", 0.0),
                })

    output_float = final_output.to(torch.float32).contiguous()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file({"output": output_float}, str(destination), metadata={
        "execution": "remote-real-expert",
        "host": args.host,
        "source_input_sha256": sha256_file(args.input),
    })
    round_trips = [sample["round_trip_ms"] for sample in samples]
    compute = [sample["worker_compute_ms"] for sample in samples]
    unattributed = [sample["unattributed_ms"] for sample in samples]
    report = {
        "output": str(destination),
        "output_sha256": sha256_file(destination),
        "host": args.host,
        "port": args.port,
        "rows": hidden.shape[0],
        "hidden_size": hidden.shape[1],
        "input_payload_bytes": len(payload),
        "output_payload_bytes": output_float.numel() * 2,
        "round_trip_mean_ms": sum(round_trips) / len(round_trips),
        "worker_compute_mean_ms": sum(compute) / len(compute),
        "unattributed_mean_ms": sum(unattributed) / len(unattributed),
        "samples": samples,
    }
    write_json(str(destination) + ".json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def run(args):
    weights = load_file(args.expert, device="cpu")
    hidden = load_file(args.input, device="cpu")["hidden_states"]
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)
    weights = {key: value.to(device=device, dtype=dtype) for key, value in weights.items()}
    hidden = hidden.to(device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(args.warmup):
            expert_forward(hidden, weights)
        if device.type == "mps":
            torch.mps.synchronize()
        samples_ms = []
        output = None
        for _ in range(args.repeats):
            started = time.perf_counter_ns()
            output = expert_forward(hidden, weights)
            if device.type == "mps":
                torch.mps.synchronize()
            samples_ms.append((time.perf_counter_ns() - started) / 1e6)

    output_cpu = output.detach().to(device="cpu", dtype=torch.float32).contiguous()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file({"output": output_cpu}, str(destination), metadata={
        "model_id": MODEL_ID,
        "revision": REVISION,
        "source_expert_sha256": sha256_file(args.expert),
        "source_input_sha256": sha256_file(args.input),
        "compute_dtype": args.dtype,
        "device": args.device,
    })
    report = {
        "output": str(destination),
        "output_sha256": sha256_file(destination),
        "expert_sha256": sha256_file(args.expert),
        "input_sha256": sha256_file(args.input),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": args.device,
        "compute_dtype": args.dtype,
        "shape": list(output_cpu.shape),
        "samples_ms": samples_ms,
        "mean_ms": sum(samples_ms) / len(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "finite": bool(torch.isfinite(output_cpu).all()),
    }
    write_json(str(destination) + ".json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


def compare(args):
    left = load_file(args.left, device="cpu")["output"].to(torch.float64)
    right = load_file(args.right, device="cpu")["output"].to(torch.float64)
    if left.shape != right.shape:
        raise ValueError("shape mismatch: %r vs %r" % (left.shape, right.shape))
    delta = left - right
    left_norm = torch.linalg.vector_norm(left)
    relative_l2 = torch.linalg.vector_norm(delta) / left_norm if left_norm else torch.tensor(math.inf)
    cosine = F.cosine_similarity(left.flatten(), right.flatten(), dim=0)
    report = {
        "left": str(args.left),
        "right": str(args.right),
        "left_sha256": sha256_file(args.left),
        "right_sha256": sha256_file(args.right),
        "shape": list(left.shape),
        "exact_equal": bool(torch.equal(left, right)),
        "allclose": bool(torch.allclose(left, right, rtol=args.rtol, atol=args.atol)),
        "rtol": args.rtol,
        "atol": args.atol,
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(relative_l2),
        "cosine_similarity": float(cosine),
    }
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["allclose"]:
        raise SystemExit(2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("extract")
    command.add_argument("--shard", required=True)
    command.add_argument("--layer", type=int, default=0)
    command.add_argument("--expert", type=int, default=0)
    command.add_argument("--output", required=True)

    command = commands.add_parser("make-input")
    command.add_argument("--rows", type=int, default=8)
    command.add_argument("--hidden-size", type=int, default=2048)
    command.add_argument("--output", required=True)

    command = commands.add_parser("run")
    command.add_argument("--expert", required=True)
    command.add_argument("--input", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    command.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    command.add_argument("--warmup", type=int, default=2)
    command.add_argument("--repeats", type=int, default=5)

    command = commands.add_parser("compare")
    command.add_argument("--left", required=True)
    command.add_argument("--right", required=True)
    command.add_argument("--rtol", type=float, default=1.6e-2)
    command.add_argument("--atol", type=float, default=1e-5)
    command.add_argument("--output")

    command = commands.add_parser("serve")
    command.add_argument("--expert", required=True)
    command.add_argument("--bind", default="0.0.0.0")
    command.add_argument("--port", type=int, default=50125)
    command.add_argument("--warmup", type=int, default=2)

    command = commands.add_parser("remote-run")
    command.add_argument("--input", required=True)
    command.add_argument("--host", required=True)
    command.add_argument("--port", type=int, default=50125)
    command.add_argument("--output", required=True)
    command.add_argument("--warmup", type=int, default=2)
    command.add_argument("--repeats", type=int, default=10)
    command.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    {"extract": extract, "make-input": make_input, "run": run, "compare": compare,
     "serve": serve_expert, "remote-run": remote_run}[args.command](args)


if __name__ == "__main__":
    main()

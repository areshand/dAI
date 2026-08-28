#!/usr/bin/env python3
"""Counterbalanced full-Qwen evaluation with local, near, and far expert paths."""

import argparse
import json
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from full_model_remote_expert import (
    MODEL_ID,
    REVISION,
    RemoteExpert,
    selected_routes,
    synchronize_device,
    tensor_metrics,
)


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values):
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def bootstrap_mean_interval(values, seed, samples=5000):
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)]
    return {
        "level": 0.95,
        "method": "paired-bootstrap-mean",
        "samples": samples,
        "low_ms": percentile(means, 0.025),
        "high_ms": percentile(means, 0.975),
    }


def parse_worker(value):
    name, separator, address = value.partition("=")
    host, port_separator, port = address.rpartition(":")
    if separator != "=" or port_separator != ":" or name not in ("near", "far") or not host:
        raise argparse.ArgumentTypeError("worker must be near=HOST:PORT or far=HOST:PORT")
    try:
        parsed_port = int(port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("worker port must be an integer") from exc
    return name, host, parsed_port


def routes_differ(reference, candidate):
    return sum(
        int(torch.any(left != right, dim=-1).sum())
        for left, right in zip(reference, candidate)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--worker", action="append", type=parse_worker, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=53)
    parser.add_argument("--prompt", default="Reply with exactly one word: hello")
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--probe-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workers = {name: (host, port) for name, host, port in args.worker}
    if set(workers) != {"near", "far"} or len(args.worker) != 2:
        parser.error("exactly one near worker and one far worker are required")
    if args.blocks < 2 or args.probe_samples < 2:
        parser.error("blocks and probe-samples must both be at least 2")

    started_at = datetime.now(timezone.utc).isoformat()
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map={"": args.device},
    )
    model.eval()
    synchronize_device(args.device)
    load_seconds = time.perf_counter() - load_started

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt")
    inputs = {key: value.to(args.device) for key, value in inputs.items()}
    local_expert = model.model.layers[args.layer].mlp.experts[args.expert]
    remotes = {
        name: RemoteExpert(host, port, args.timeout)
        for name, (host, port) in workers.items()
    }

    hidden_size = int(model.config.hidden_size)
    probe_hidden = torch.zeros((7, hidden_size), dtype=torch.bfloat16, device=args.device)
    discovery = {}
    for name, remote in remotes.items():
        remote(probe_hidden)
        remote.samples.clear()
        for _ in range(args.probe_samples):
            remote(probe_hidden)
        samples = list(remote.samples)
        discovery[name] = {
            "samples": len(samples),
            "rows": int(probe_hidden.shape[0]),
            "total_boundary_ms": distribution([item["total_boundary_ms"] for item in samples]),
            "round_trip_ms": distribution([item["round_trip_ms"] for item in samples]),
            "worker_compute_ms": distribution([item["worker_compute_ms"] for item in samples]),
            "unattributed_ms": distribution([item["unattributed_ms"] for item in samples]),
        }
        remote.samples.clear()
    selected_worker = min(discovery, key=lambda name: discovery[name]["total_boundary_ms"]["p50"])

    def run(path, measured, block=None, order_index=None):
        model.model.layers[args.layer].mlp.experts[args.expert] = (
            local_expert if path == "local" else remotes[path]
        )
        before = {name: len(remote.samples) for name, remote in remotes.items()}
        run_started = time.perf_counter()
        with torch.no_grad():
            result = model(**inputs, use_cache=False, output_router_logits=True, return_dict=True)
        synchronize_device(args.device)
        elapsed_ms = (time.perf_counter() - run_started) * 1000
        logits = result.logits[:, -1, :].detach().cpu().float()
        routes = selected_routes(result.router_logits, model.config.num_experts_per_tok)
        token_id = int(torch.argmax(logits, dim=-1)[0])
        calls = {
            name: remote.samples[before[name]:]
            for name, remote in remotes.items()
        }
        if path != "local" and not calls[path]:
            raise RuntimeError("the selected expert was not naturally routed to during %s" % path)
        if any(calls[name] for name in calls if name != path):
            raise RuntimeError("an inactive remote worker received an expert call")
        return {
            "path": path,
            "measured": measured,
            "block": block,
            "order_index": order_index,
            "elapsed_ms": elapsed_ms,
            "token_id": token_id,
            "remote_calls": calls[path] if path != "local" else [],
            "logits": logits,
            "routes": routes,
        }

    rng = random.Random(args.seed)
    measured_runs = []
    block_orders = []
    try:
        warmups = {path: run(path, False) for path in ("local", "near", "far")}
        reference_logits = warmups["local"]["logits"]
        reference_routes = warmups["local"]["routes"]
        reference_token = warmups["local"]["token_id"]
        for block in range(args.blocks):
            order = ["local", "near", "far"]
            rng.shuffle(order)
            block_orders.append(order)
            for order_index, path in enumerate(order):
                measured_runs.append(run(path, True, block, order_index))
    finally:
        for remote in remotes.values():
            remote.close()

    serialized_runs = []
    for item in measured_runs:
        serialized_runs.append({
            key: value for key, value in item.items() if key not in ("logits", "routes")
        } | {
            "route_differing_token_layers": routes_differ(reference_routes, item["routes"]),
            "logits_metrics": tensor_metrics(reference_logits, item["logits"]),
            "token_equal": item["token_id"] == reference_token,
        })

    by_path = {
        path: [item for item in serialized_runs if item["path"] == path]
        for path in ("local", "near", "far")
    }
    paired_deltas = {}
    for left, right in (("local", "near"), ("local", "far"), ("near", "far")):
        deltas = []
        for block in range(args.blocks):
            left_run = next(item for item in by_path[left] if item["block"] == block)
            right_run = next(item for item in by_path[right] if item["block"] == block)
            deltas.append(right_run["elapsed_ms"] - left_run["elapsed_ms"])
        paired_deltas["%s_to_%s" % (left, right)] = {
            "delta_ms": distribution(deltas),
            "mean_interval": bootstrap_mean_interval(deltas, args.seed + len(paired_deltas) + 1),
        }

    path_summary = {}
    for path, runs in by_path.items():
        calls = [call for item in runs for call in item["remote_calls"]]
        path_summary[path] = {"forward_ms": distribution([item["elapsed_ms"] for item in runs])}
        if calls:
            path_summary[path]["remote_call_count"] = len(calls)
            path_summary[path]["remote_boundary_ms"] = distribution(
                [item["total_boundary_ms"] for item in calls]
            )
            path_summary[path]["worker_compute_ms"] = distribution(
                [item["worker_compute_ms"] for item in calls]
            )

    report = {
        "schema": "qwen3-full-model-multi-worker-placement.v1",
        "claim_scope": "real full-model forward with one naturally routed expert movable across two workers",
        "started_at": started_at,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "runtime": {
            "device": args.device,
            "dtype": "bfloat16",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "parameters": {
            "prompt": args.prompt,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "layer": args.layer,
            "expert": args.expert,
            "blocks": args.blocks,
            "probe_samples": args.probe_samples,
            "seed": args.seed,
        },
        "load_seconds": load_seconds,
        "discovery": discovery,
        "latency_aware_worker": selected_worker,
        "block_orders": block_orders,
        "warmup_ms": {path: item["elapsed_ms"] for path, item in warmups.items()},
        "runs": serialized_runs,
        "summary": {
            "paths": path_summary,
            "paired_deltas": paired_deltas,
            "all_routes_equal": all(item["route_differing_token_layers"] == 0 for item in serialized_runs),
            "all_logits_allclose": all(item["logits_metrics"]["allclose"] for item in serialized_runs),
            "all_tokens_equal": all(item["token_equal"] for item in serialized_runs),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "latency_aware_worker": selected_worker,
        "summary": report["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

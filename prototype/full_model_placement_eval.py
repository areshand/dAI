#!/usr/bin/env python3
"""Balanced paired local-only versus remote-only full-forward evaluation."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from full_model_remote_expert import RemoteExpert, selected_routes, tensor_metrics


MODEL_ID = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_mean_interval(values, seed, samples=5000):
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)]
    return {"level": 0.95, "method": "paired-bootstrap-mean", "samples": samples,
            "low_ms": percentile(means, 0.025), "high_ms": percentile(means, 0.975)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=50126)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--prompt", default="Reply with exactly one word: hello")
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        device_map={"": "mps"},
    )
    model.eval()
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - load_started

    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt")
    inputs = {key: value.to("mps") for key, value in inputs.items()}
    local_expert = model.model.layers[args.layer].mlp.experts[args.expert]
    remote_expert = RemoteExpert(args.host, args.port, args.timeout)

    def run(path, measured, pair_id=None, order_index=None):
        model.model.layers[args.layer].mlp.experts[args.expert] = (
            local_expert if path == "local" else remote_expert
        )
        remote_before = len(remote_expert.samples)
        started = time.perf_counter()
        with torch.no_grad():
            result = model(**inputs, use_cache=False, output_router_logits=True, return_dict=True)
        torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        logits = result.logits[:, -1, :].detach().cpu().float()
        routes = selected_routes(result.router_logits, model.config.num_experts_per_tok)
        token_id = int(torch.argmax(logits, dim=-1)[0])
        remote_samples = remote_expert.samples[remote_before:]
        return {
            "path": path,
            "measured": measured,
            "pair_id": pair_id,
            "order_index": order_index,
            "elapsed_ms": elapsed_ms,
            "token_id": token_id,
            "logits": logits,
            "routes": routes,
            "remote_calls": remote_samples,
        }

    local_warm = run("local", False)
    remote_warm = run("remote", False)
    reference_logits = local_warm["logits"]
    reference_routes = local_warm["routes"]
    reference_token = local_warm["token_id"]

    rng = random.Random(args.seed)
    runs = []
    pairs = []
    try:
        for pair_id in range(args.pairs):
            order = ["local", "remote"]
            rng.shuffle(order)
            pair_runs = [run(path, True, pair_id, index) for index, path in enumerate(order)]
            by_path = {item["path"]: item for item in pair_runs}
            delta = by_path["remote"]["elapsed_ms"] - by_path["local"]["elapsed_ms"]
            pairs.append({"pair_id": pair_id, "order": order, "local_ms": by_path["local"]["elapsed_ms"],
                          "remote_ms": by_path["remote"]["elapsed_ms"], "delta_ms": delta})
            runs.extend(pair_runs)
    finally:
        remote_expert.close()

    serialized_runs = []
    for item in runs:
        differing_rows = sum(int(torch.any(left != right, dim=-1).sum())
                             for left, right in zip(reference_routes, item["routes"]))
        serialized_runs.append({
            key: value for key, value in item.items() if key not in ("logits", "routes")
        } | {
            "route_differing_token_layers": differing_rows,
            "logits_metrics": tensor_metrics(reference_logits, item["logits"]),
            "token_equal": item["token_id"] == reference_token,
        })

    deltas = [pair["delta_ms"] for pair in pairs]
    local_times = [pair["local_ms"] for pair in pairs]
    remote_times = [pair["remote_ms"] for pair in pairs]
    remote_calls = [call for item in serialized_runs for call in item["remote_calls"]]
    report = {
        "schema": "qwen3-paired-placement-eval.v1",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "prompt": args.prompt,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "remote_layer": args.layer,
        "remote_expert": args.expert,
        "pairs": args.pairs,
        "seed": args.seed,
        "load_seconds": load_seconds,
        "warmup": {"local_ms": local_warm["elapsed_ms"], "remote_ms": remote_warm["elapsed_ms"]},
        "paired_results": pairs,
        "runs": serialized_runs,
        "summary": {
            "local_mean_ms": statistics.fmean(local_times),
            "local_median_ms": statistics.median(local_times),
            "remote_mean_ms": statistics.fmean(remote_times),
            "remote_median_ms": statistics.median(remote_times),
            "delta_mean_ms": statistics.fmean(deltas),
            "delta_median_ms": statistics.median(deltas),
            "delta_min_ms": min(deltas),
            "delta_max_ms": max(deltas),
            "delta_mean_interval": bootstrap_mean_interval(deltas, args.seed + 1),
            "all_routes_equal": all(item["route_differing_token_layers"] == 0 for item in serialized_runs),
            "all_logits_allclose": all(item["logits_metrics"]["allclose"] for item in serialized_runs),
            "all_tokens_equal": all(item["token_equal"] for item in serialized_runs),
            "remote_call_count": len(remote_calls),
            "remote_boundary_mean_ms": statistics.fmean(call["total_boundary_ms"] for call in remote_calls),
            "remote_round_trip_mean_ms": statistics.fmean(call["round_trip_ms"] for call in remote_calls),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": report["summary"],
                      "paired_results": pairs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

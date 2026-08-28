#!/usr/bin/env python3
"""Compare full Qwen3 with one local versus one remote expert."""

import argparse
import json
import socket
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from real_expert_probe import bf16_from_bytes, recv_packet, send_packet, tensor_to_bytes


MODEL_ID = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


class RemoteExpert(torch.nn.Module):
    def __init__(self, host, port, timeout):
        super().__init__()
        self.connection = socket.create_connection((host, port), timeout=timeout)
        self.connection.settimeout(timeout)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.samples = []
        self.next_request_id = 0

    def forward(self, hidden):
        if hidden.shape[0] == 0:
            return hidden.new_empty((0, hidden.shape[1]))
        boundary_started = time.perf_counter_ns()
        stage_started = boundary_started
        cpu_hidden = hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        payload = tensor_to_bytes(cpu_hidden)
        host_stage_ms = (time.perf_counter_ns() - stage_started) / 1e6
        request_id = self.next_request_id
        self.next_request_id += 1
        started = time.perf_counter_ns()
        send_packet(self.connection, {
            "op": "expert_forward",
            "request_id": request_id,
            "rows": cpu_hidden.shape[0],
            "hidden_size": cpu_hidden.shape[1],
            "dtype": "bfloat16",
        }, payload)
        response, output_bytes = recv_packet(self.connection)
        round_trip_ms = (time.perf_counter_ns() - started) / 1e6
        output = bf16_from_bytes(output_bytes, response["rows"], response["hidden_size"])
        restore_started = time.perf_counter_ns()
        restored = output.to(device=hidden.device, dtype=hidden.dtype)
        synchronize_device(hidden.device)
        restore_ms = (time.perf_counter_ns() - restore_started) / 1e6
        self.samples.append({
            "request_id": request_id,
            "rows": cpu_hidden.shape[0],
            "input_bytes": len(payload),
            "output_bytes": len(output_bytes),
            "round_trip_ms": round_trip_ms,
            "host_stage_ms": host_stage_ms,
            "restore_ms": restore_ms,
            "total_boundary_ms": (time.perf_counter_ns() - boundary_started) / 1e6,
            "worker_compute_ms": response["worker_compute_ms"],
            "worker_preprocess_ms": response.get("worker_preprocess_ms", 0.0),
            "worker_serialize_ms": response.get("worker_serialize_ms", 0.0),
            "unattributed_ms": round_trip_ms - response["worker_compute_ms"]
                - response.get("worker_preprocess_ms", 0.0)
                - response.get("worker_serialize_ms", 0.0),
        })
        return restored

    def close(self):
        self.connection.close()


def synchronize_device(device):
    """Wait for asynchronous accelerator work without assuming Apple MPS."""
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def selected_routes(router_logits, top_k):
    return [torch.topk(layer, k=top_k, dim=-1).indices.cpu() for layer in router_logits]


def tensor_metrics(left, right, rtol=1.6e-2, atol=1e-5):
    left = left.to(torch.float64)
    right = right.to(torch.float64)
    delta = left - right
    return {
        "exact_equal": bool(torch.equal(left, right)),
        "allclose": bool(torch.allclose(left, right, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(left)),
        "cosine_similarity": float(F.cosine_similarity(left.flatten(), right.flatten(), dim=0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=50126)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--prompt", default="Reply with exactly one word: hello")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
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
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt")
    inputs = {key: value.to("mps") for key, value in inputs.items()}

    baseline_started = time.perf_counter()
    with torch.no_grad():
        baseline = model(**inputs, use_cache=False, output_router_logits=True, return_dict=True)
    torch.mps.synchronize()
    baseline_seconds = time.perf_counter() - baseline_started
    baseline_logits = baseline.logits[:, -1, :].detach().cpu().float()
    baseline_routes = selected_routes(baseline.router_logits, model.config.num_experts_per_tok)
    baseline_token = int(torch.argmax(baseline_logits, dim=-1)[0])
    del baseline

    remote = RemoteExpert(args.host, args.port, args.timeout)
    model.model.layers[args.layer].mlp.experts[args.expert] = remote
    distributed_started = time.perf_counter()
    try:
        with torch.no_grad():
            distributed = model(**inputs, use_cache=False, output_router_logits=True, return_dict=True)
        torch.mps.synchronize()
        distributed_seconds = time.perf_counter() - distributed_started
        distributed_logits = distributed.logits[:, -1, :].detach().cpu().float()
        distributed_routes = selected_routes(distributed.router_logits, model.config.num_experts_per_tok)
        distributed_token = int(torch.argmax(distributed_logits, dim=-1)[0])
    finally:
        remote.close()

    route_differences = []
    differing_token_layers = 0
    for layer_id, (left, right) in enumerate(zip(baseline_routes, distributed_routes)):
        different_rows = torch.any(left != right, dim=-1)
        count = int(different_rows.sum())
        differing_token_layers += count
        if count:
            route_differences.append({"layer_id": layer_id, "different_token_rows": count})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tensors_path = output.with_suffix(".safetensors")
    save_file({"baseline_last_logits": baseline_logits.contiguous(),
               "distributed_last_logits": distributed_logits.contiguous()}, str(tensors_path))
    report = {
        "schema": "qwen3-one-remote-expert-full-forward.v1",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "dtype": "bfloat16",
        "local_device": "mps",
        "remote_device": "cpu",
        "remote_host": args.host,
        "remote_port": args.port,
        "remote_layer": args.layer,
        "remote_expert": args.expert,
        "prompt": args.prompt,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "load_seconds": load_seconds,
        "baseline_forward_seconds": baseline_seconds,
        "distributed_forward_seconds": distributed_seconds,
        "forward_delta_ms": (distributed_seconds - baseline_seconds) * 1000,
        "remote_calls": remote.samples,
        "remote_call_count": len(remote.samples),
        "remote_rows": sum(sample["rows"] for sample in remote.samples),
        "route_equality": {
            "exact": differing_token_layers == 0,
            "differing_token_layers": differing_token_layers,
            "differences": route_differences,
        },
        "last_logits": tensor_metrics(baseline_logits, distributed_logits),
        "baseline_greedy_token_id": baseline_token,
        "distributed_greedy_token_id": distributed_token,
        "greedy_token_equal": baseline_token == distributed_token,
        "greedy_text": tokenizer.decode([distributed_token], skip_special_tokens=False),
        "logits_artifact": str(tensors_path),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run local oracle and remote Qwen expert concurrently during generation."""

import argparse
import concurrent.futures
import hashlib
import json
import socket
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from real_expert_probe import bf16_from_bytes, recv_packet, send_packet, tensor_to_bytes


MODEL_ID = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
DEFAULT_PROMPTS = [
    "Reply with exactly one word: hello",
    "What is 2 + 2? Reply with only the number.",
    "Name the capital of France in one word.",
]


def output_metrics(local_output, remote_output):
    left = local_output.detach().to(device="cpu", dtype=torch.float32)
    right = remote_output.to(torch.float32)
    delta = left.to(torch.float64) - right.to(torch.float64)
    left64 = left.to(torch.float64)
    return {
        "exact_equal": bool(torch.equal(left, right)),
        "allclose": bool(torch.allclose(left, right, rtol=1.6e-2, atol=1e-5)),
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.abs().mean()) if delta.numel() else 0.0,
        "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(left64))
            if delta.numel() and torch.linalg.vector_norm(left64) else 0.0,
        "cosine_similarity": float(F.cosine_similarity(left64.flatten(), right.to(torch.float64).flatten(), dim=0))
            if delta.numel() else 1.0,
    }


class ShadowExpert(torch.nn.Module):
    def __init__(self, local_expert, host, port, timeout):
        super().__init__()
        self.local_expert = local_expert
        self.connection = socket.create_connection((host, port), timeout=timeout)
        self.connection.settimeout(timeout)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.samples = []
        self.next_request_id = 0
        self.context = {}

    def _remote(self, common_start_ns, request_id, rows, hidden_size, payload):
        task_start_ns = time.perf_counter_ns()
        send_packet(self.connection, {
            "op": "expert_forward",
            "request_id": request_id,
            "rows": rows,
            "hidden_size": hidden_size,
            "dtype": "bfloat16",
        }, payload)
        response, output_bytes = recv_packet(self.connection)
        finished_ns = time.perf_counter_ns()
        output = bf16_from_bytes(output_bytes, response["rows"], response["hidden_size"])
        return {
            "output": output,
            "response": response,
            "task_start_skew_ms": (task_start_ns - common_start_ns) / 1e6,
            "completion_ms": (finished_ns - common_start_ns) / 1e6,
            "output_bytes": len(output_bytes),
        }

    def forward(self, hidden):
        if hidden.shape[0] == 0:
            return self.local_expert(hidden)

        # Shadow validation branches after one shared host-staging operation.
        stage_started = time.perf_counter_ns()
        cpu_hidden = hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        payload = tensor_to_bytes(cpu_hidden)
        stage_ms = (time.perf_counter_ns() - stage_started) / 1e6
        request_id = self.next_request_id
        self.next_request_id += 1

        common_start_ns = time.perf_counter_ns()
        future = self.executor.submit(
            self._remote, common_start_ns, request_id, hidden.shape[0], hidden.shape[1], payload
        )
        local_started_ns = time.perf_counter_ns()
        local_output = self.local_expert(hidden)
        torch.mps.synchronize()
        local_completion_ms = (time.perf_counter_ns() - common_start_ns) / 1e6
        remote = future.result()
        response = remote["response"]
        metrics = output_metrics(local_output, remote["output"])
        self.samples.append({
            **self.context,
            "request_id": request_id,
            "rows": hidden.shape[0],
            "input_bytes": len(payload),
            "input_sha256": hashlib.sha256(payload).hexdigest(),
            "host_stage_ms": stage_ms,
            "local_start_skew_ms": (local_started_ns - common_start_ns) / 1e6,
            "local_completion_ms": local_completion_ms,
            "remote_task_start_skew_ms": remote["task_start_skew_ms"],
            "remote_completion_ms": remote["completion_ms"],
            "remote_output_bytes": remote["output_bytes"],
            "worker_preprocess_ms": response.get("worker_preprocess_ms", 0.0),
            "worker_compute_ms": response["worker_compute_ms"],
            "worker_serialize_ms": response.get("worker_serialize_ms", 0.0),
            "output_metrics": metrics,
        })
        # Oracle output remains authoritative in shadow mode.
        return local_output

    def close(self):
        self.executor.shutdown(wait=True)
        self.connection.close()


def generate(model, tokenizer, prompt, max_new_tokens):
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt")
    inputs = {key: value.to("mps") for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            use_cache=True,
        )
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    new_ids = generated[0, inputs["input_ids"].shape[1]:].detach().cpu().tolist()
    return {
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "new_token_ids": new_ids,
        "new_text": tokenizer.decode(new_ids, skip_special_tokens=False),
        "seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=50126)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--expert", type=int, required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--shadow-repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prompts = args.prompt or DEFAULT_PROMPTS

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

    baselines = []
    for prompt_id, prompt in enumerate(prompts):
        baseline = generate(model, tokenizer, prompt, args.max_new_tokens)
        baselines.append({"prompt_id": prompt_id, "prompt": prompt, **baseline})

    local_expert = model.model.layers[args.layer].mlp.experts[args.expert]
    shadow = ShadowExpert(local_expert, args.host, args.port, args.timeout)
    model.model.layers[args.layer].mlp.experts[args.expert] = shadow
    shadow_runs = []
    try:
        for repeat in range(args.shadow_repeats):
            for prompt_id, prompt in enumerate(prompts):
                shadow.context = {"repeat": repeat, "prompt_id": prompt_id}
                observed = generate(model, tokenizer, prompt, args.max_new_tokens)
                baseline = baselines[prompt_id]
                shadow_runs.append({
                    "repeat": repeat,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    **observed,
                    "baseline_token_ids": baseline["new_token_ids"],
                    "token_ids_equal": observed["new_token_ids"] == baseline["new_token_ids"],
                })
    finally:
        shadow.close()

    report = {
        "schema": "qwen3-shadow-expert-generation.v1",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "dtype": "bfloat16",
        "oracle_device": "mps",
        "remote_device": "cpu",
        "remote_host": args.host,
        "remote_layer": args.layer,
        "remote_expert": args.expert,
        "load_seconds": load_seconds,
        "max_new_tokens": args.max_new_tokens,
        "shadow_repeats": args.shadow_repeats,
        "baselines": baselines,
        "shadow_runs": shadow_runs,
        "expert_calls": shadow.samples,
        "expert_call_count": len(shadow.samples),
        "covered_prompts": sorted(set(sample["prompt_id"] for sample in shadow.samples)),
        "all_expert_outputs_allclose": all(sample["output_metrics"]["allclose"] for sample in shadow.samples),
        "all_expert_outputs_exact": all(sample["output_metrics"]["exact_equal"] for sample in shadow.samples),
        "all_generated_tokens_equal": all(run["token_ids_equal"] for run in shadow_runs),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "load_seconds": load_seconds,
        "expert_call_count": report["expert_call_count"],
        "covered_prompts": report["covered_prompts"],
        "all_expert_outputs_allclose": report["all_expert_outputs_allclose"],
        "all_expert_outputs_exact": report["all_expert_outputs_exact"],
        "all_generated_tokens_equal": report["all_generated_tokens_equal"],
        "shadow_runs": shadow_runs,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

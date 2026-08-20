#!/usr/bin/env python3
"""Load the pinned full Qwen3 MoE checkpoint on MPS and generate one token."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-30B-A3B"
REVISION = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def mps_memory():
    result = {}
    for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
        function = getattr(torch.mps, name, None)
        if function is not None:
            try:
                result[name + "_bytes"] = int(function())
            except Exception as exc:
                result[name + "_error"] = repr(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="Reply with exactly one word: hello")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--capture-routes", action="store_true")
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    process = psutil.Process()
    started_at = datetime.now(timezone.utc).isoformat()
    rss_before = process.memory_info().rss
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
    memory_after_load = mps_memory()
    rss_after_load = process.memory_info().rss

    messages = [{"role": "user", "content": args.prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt")
    inputs = {key: value.to("mps") for key, value in inputs.items()}

    route_capture = None
    if args.capture_routes:
        route_started = time.perf_counter()
        with torch.no_grad():
            routed = model(
                **inputs,
                use_cache=False,
                output_router_logits=True,
                return_dict=True,
            )
        torch.mps.synchronize()
        layers = []
        for layer_id, logits in enumerate(routed.router_logits):
            selected = torch.topk(logits, k=model.config.num_experts_per_tok, dim=-1).indices.cpu()
            unique, counts = torch.unique(selected, return_counts=True)
            layers.append({
                "layer_id": layer_id,
                "last_token_selected_experts": selected[-1].tolist(),
                "prompt_expert_counts": {str(int(key)): int(value)
                                         for key, value in zip(unique, counts)},
            })
        route_capture = {
            "seconds": time.perf_counter() - route_started,
            "layers": layers,
        }
        del routed

    generation_started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.mps.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    new_ids = generated[0, inputs["input_ids"].shape[1]:].detach().cpu()
    text = tokenizer.decode(new_ids, skip_special_tokens=False)

    report = {
        "schema": "qwen3-full-model-smoke.v1",
        "started_at": started_at,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "model_path": args.model_path,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dtype": "bfloat16",
        "device": "mps",
        "attention": "eager",
        "prompt": args.prompt,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "new_token_ids": new_ids.tolist(),
        "new_text": text,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "seconds_per_generated_token": generation_seconds / max(1, len(new_ids)),
        "rss_before_bytes": rss_before,
        "rss_after_load_bytes": rss_after_load,
        "mps_after_load": memory_after_load,
        "mps_after_generation": mps_memory(),
        "route_capture": route_capture,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

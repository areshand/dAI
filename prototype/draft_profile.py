#!/usr/bin/env python3
"""Profile steady-state autoregressive draft steps on one hardware class."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from generation_benchmark import distribution, exact_token_prompt


def synchronize(torch, device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--round-sizes", default="2,4,8,16")
    parser.add_argument("--nonce", default="dai-draft-profile-v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(args.device)
    model.eval()
    prompt, prompt_ids = exact_token_prompt(tokenizer, args.prompt_tokens, args.nonce)
    input_ids = torch.tensor([prompt_ids], device=args.device)
    round_sizes = sorted({int(value) for value in args.round_sizes.split(",")})
    if not round_sizes or min(round_sizes) < 1:
        raise ValueError("round sizes must be positive")
    maximum_steps = max(round_sizes)

    repetitions = []
    with torch.inference_mode():
        for index in range(args.warmups + args.repetitions):
            synchronize(torch, args.device)
            prefill_started = time.perf_counter()
            output = model(input_ids=input_ids, use_cache=True)
            synchronize(torch, args.device)
            prefill_seconds = time.perf_counter() - prefill_started
            past = output.past_key_values
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            step_seconds = []
            generated_ids = []
            for _ in range(maximum_steps):
                synchronize(torch, args.device)
                step_started = time.perf_counter()
                output = model(input_ids=next_token, past_key_values=past, use_cache=True)
                synchronize(torch, args.device)
                step_seconds.append(time.perf_counter() - step_started)
                generated_ids.append(int(next_token.item()))
                past = output.past_key_values
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            repetitions.append({
                "index": index,
                "measured": index >= args.warmups,
                "prefill_seconds": prefill_seconds,
                "step_seconds": step_seconds,
                "round_seconds": {
                    str(size): sum(step_seconds[:size]) for size in round_sizes
                },
                "generated_token_ids": generated_ids,
            })

    measured = [item for item in repetitions if item["measured"]]
    result = {
        "schema": "dai-draft-profile.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "prompt_tokens": len(prompt_ids),
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "round_sizes": round_sizes,
        "summary": {
            "prefill_seconds": distribution([item["prefill_seconds"] for item in measured]),
            "decode_step_seconds": distribution([
                value for item in measured for value in item["step_seconds"]
            ]),
            "round_seconds": {
                str(size): distribution([
                    item["round_seconds"][str(size)] for item in measured
                ]) for size in round_sizes
            },
            "mean_decode_tokens_per_second": 1.0 / statistics.fmean([
                value for item in measured for value in item["step_seconds"]
            ]),
        },
        "runs": repetitions,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **result["summary"]}, indent=2))


if __name__ == "__main__":
    main()

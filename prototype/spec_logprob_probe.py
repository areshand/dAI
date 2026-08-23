#!/usr/bin/env python3
"""Capture greedy token IDs and target logprobs for spec-parity diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from generation_benchmark import exact_token_prompt


def generate(endpoint: str, prompt: str, max_tokens: int, top_logprobs: int) -> dict:
    payload = json.dumps(
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "return_text_in_logprobs": False,
            "top_logprobs_num": top_logprobs,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--top-logprobs", type=int, default=10)
    parser.add_argument("--nonce", default="dai-generation-v2")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    source = Path(args.prompt_file).read_text(encoding="utf-8")
    prompt, prompt_ids = exact_token_prompt(
        tokenizer, args.prompt_tokens, args.nonce, source
    )
    response = generate(args.endpoint, prompt, args.max_tokens, args.top_logprobs)
    meta = response.get("meta_info") or {}
    token_logprobs = meta.get("output_token_logprobs") or []
    top_logprobs = meta.get("output_top_logprobs") or []
    token_ids = [int(item[1]) for item in token_logprobs]
    if len(token_ids) != args.max_tokens or len(top_logprobs) != len(token_ids):
        raise RuntimeError(
            "server returned incomplete logprobs: "
            f"tokens={len(token_ids)} top_logprobs={len(top_logprobs)}"
        )

    report = {
        "schema": "dai-spec-logprob-probe.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "prompt_tokens": len(prompt_ids),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "max_tokens": args.max_tokens,
        "output_token_ids": token_ids,
        "output_token_logprobs": token_logprobs,
        "output_top_logprobs": top_logprobs,
        "spec_tokens_details": {
            key: value for key, value in meta.items() if key.startswith("spec_")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "variant": args.variant,
                "output_token_sha256": hashlib.sha256(
                    json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "spec_tokens_details": report["spec_tokens_details"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

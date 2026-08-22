#!/usr/bin/env python3
"""Capture request-aligned SGLang speculative-decoding counters.

The pinned SGLang v0.5.16 OpenAI-compatible response omits these counters, but
its native `/generate` response retains them in `meta_info`.  This probe is a
diagnostic companion to the streaming benchmark, not a replacement for it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from generation_benchmark import exact_token_prompt


SPEC_KEYS = (
    "spec_accept_rate",
    "spec_accept_length",
    "spec_num_correct_drafts",
    "spec_num_proposed_drafts",
    "spec_verify_ct",
    "spec_correct_drafts_histogram",
)


def extract_spec_details(response: dict) -> dict:
    meta = response.get("meta_info") or {}
    return {key: meta[key] for key in SPEC_KEYS if key in meta}


def generate(endpoint: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    payload = json.dumps(
        {
            "text": prompt,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_tokens,
                "ignore_eos": True,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return {"body": body, "total_seconds": time.perf_counter() - started}


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-tokens", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--nonce", default="dai-spec-probe-v1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    source = Path(args.prompt_file).read_text() if args.prompt_file else None
    prompt, prompt_ids = exact_token_prompt(
        tokenizer, args.prompt_tokens, args.nonce, source
    )
    runs = []
    for index in range(args.warmups + args.repetitions):
        result = generate(args.endpoint, prompt, args.max_tokens, args.timeout)
        body = result.pop("body")
        text = body.get("text") or ""
        output_ids = tokenizer.encode(text, add_special_tokens=False)
        details = extract_spec_details(body)
        runs.append(
            {
                "index": index,
                "measured": index >= args.warmups,
                **result,
                "output_tokens": len(output_ids),
                "output_token_sha256": hashlib.sha256(
                    json.dumps(output_ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "spec_tokens_details": details,
            }
        )
    measured = [run for run in runs if run["measured"]]
    missing = sum(not run["spec_tokens_details"] for run in measured)
    report = {
        "schema": "dai-sglang-native-spec-probe.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "prompt_tokens": len(prompt_ids),
        "max_tokens": args.max_tokens,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "missing_spec_detail_runs": missing,
        "unique_output_hashes": len(
            {run["output_token_sha256"] for run in measured}
        ),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "runs"}, indent=2))
    if missing:
        raise RuntimeError(
            f"pinned server omitted speculative details for {missing} measured runs"
        )


if __name__ == "__main__":
    main()

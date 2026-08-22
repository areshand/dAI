#!/usr/bin/env python3
"""Benchmark a warmed OpenAI-compatible completion server with an exact token prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SEED_TEXT = (
    "Decentralized inference coordinates attention, routing, expert execution, "
    "network transfer, verification, caching, and scheduling across heterogeneous "
    "workers. The benchmark uses a unique uncached prompt and deterministic decoding. "
)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def streaming_trace(events: list[dict], tokenizer) -> dict:
    """Map SSE arrivals to token arrivals without inventing sub-event timing.

    Tokenizers can revise the final token at text-piece boundaries.  A trace is
    therefore declared token-exact only when every cumulative encode adds
    exactly one token, never regresses, and the final token count equals the
    number of non-empty SSE events.  Coalesced/ambiguous traces retain their raw
    event inter-arrivals but do not produce a client-token ITL distribution.
    """

    cumulative_text = ""
    previous_count = 0
    arrivals: list[float] = []
    event_token_deltas: list[int] = []
    cumulative_token_counts: list[int] = []
    regressions = 0
    for event in events:
        cumulative_text += event["text"]
        count = len(tokenizer.encode(cumulative_text, add_special_tokens=False))
        delta = count - previous_count
        if delta < 0:
            regressions += 1
        event_token_deltas.append(delta)
        cumulative_token_counts.append(count)
        if delta > 0:
            arrivals.extend([float(event["arrival_seconds"])] * delta)
        previous_count = count

    event_arrivals = [float(event["arrival_seconds"]) for event in events]
    event_interarrivals = [
        later - earlier for earlier, later in zip(event_arrivals, event_arrivals[1:])
    ]
    token_exact = bool(events) and regressions == 0 and all(
        delta == 1 for delta in event_token_deltas
    )
    token_itl = [
        later - earlier for earlier, later in zip(arrivals, arrivals[1:])
    ] if token_exact else []
    return {
        "event_arrival_seconds": event_arrivals,
        "event_interarrival_seconds": event_interarrivals,
        "event_token_deltas": event_token_deltas,
        "cumulative_token_counts": cumulative_token_counts,
        "tokenization_regressions": regressions,
        "coalesced_event_count": sum(delta > 1 for delta in event_token_deltas),
        "ambiguous_event_count": sum(delta != 1 for delta in event_token_deltas),
        "client_token_itl_valid": token_exact,
        "client_token_arrival_seconds": arrivals if token_exact else [],
        "client_token_itl_seconds": token_itl,
    }


def exact_token_prompt(
    tokenizer, target_tokens: int, nonce: str, source_text: str | None = None
) -> tuple[str, list[int]]:
    if target_tokens < 32:
        raise ValueError("target_tokens must be at least 32")
    text = f"Run nonce {nonce}. " + (source_text or DEFAULT_SEED_TEXT)
    if source_text is not None and len(
        tokenizer.encode(text, add_special_tokens=False)
    ) < target_tokens + 8:
        raise ValueError("prompt source does not contain enough tokens")
    while source_text is None and len(
        tokenizer.encode(text, add_special_tokens=False)
    ) < target_tokens + 8:
        text += DEFAULT_SEED_TEXT
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    prompt = tokenizer.decode(token_ids, skip_special_tokens=False)
    round_trip = tokenizer.encode(prompt, add_special_tokens=False)
    if len(round_trip) != target_tokens:
        raise RuntimeError(
            f"tokenizer decode/encode round trip changed prompt length: "
            f"wanted {target_tokens}, got {len(round_trip)}"
        )
    return prompt, round_trip


def stream_completion(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_text_at = None
    pieces: list[str] = []
    stream_events: list[dict[str, float | str]] = []
    usage = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                piece = choices[0].get("text") or ""
                if piece:
                    arrived_at = time.perf_counter()
                    if first_text_at is None:
                        first_text_at = arrived_at
                    pieces.append(piece)
                    stream_events.append({
                        "arrival_seconds": arrived_at - started,
                        "text": piece,
                    })
    finished = time.perf_counter()
    if first_text_at is None:
        first_text_at = finished
    return {
        "text": "".join(pieces),
        "usage": usage,
        "ttft_seconds": first_text_at - started,
        "total_seconds": finished - started,
        "stream_seconds": max(0.0, finished - first_text_at),
        "stream_events": stream_events,
    }


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True, help="Model name sent to the server")
    parser.add_argument("--tokenizer", required=True, help="Local tokenizer path or model ID")
    parser.add_argument("--variant", required=True, help="Human-readable server configuration")
    parser.add_argument("--prompt-tokens", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--nonce", default="dai-generation-v1")
    parser.add_argument(
        "--prompt-file",
        help="Use diverse UTF-8 source text instead of the synthetic repeated seed",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    source_text = (
        Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else None
    )
    prompt, prompt_ids = exact_token_prompt(
        tokenizer, args.prompt_tokens, args.nonce, source_text
    )
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    runs = []
    for index in range(args.warmups + args.repetitions):
        result = stream_completion(
            args.endpoint, args.model, prompt, args.max_tokens, args.timeout
        )
        trace = streaming_trace(result.pop("stream_events"), tokenizer)
        output_ids = tokenizer.encode(result.pop("text"), add_special_tokens=False)
        output_count = int((result.get("usage") or {}).get("completion_tokens") or len(output_ids))
        decode_seconds = result["stream_seconds"]
        output_tps = (
            max(0, output_count - 1) / decode_seconds
            if decode_seconds > 0 and output_count > 1
            else 0.0
        )
        runs.append({
            "index": index,
            "measured": index >= args.warmups,
            **result,
            "stream_trace": trace,
            "output_tokens": output_count,
            "output_tps": output_tps,
            "output_token_sha256": hashlib.sha256(
                json.dumps(output_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "output_token_ids": output_ids,
        })

    measured = [run for run in runs if run["measured"]]
    report = {
        "schema": "dai-openai-generation-benchmark.v2",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint,
        "model": args.model,
        "variant": args.variant,
        "prompt_tokens": len(prompt_ids),
        "prompt_sha256": prompt_sha256,
        "prompt_source": "file" if args.prompt_file else "synthetic-repeat",
        "max_tokens": args.max_tokens,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "summary": {
            "ttft_seconds": distribution([run["ttft_seconds"] for run in measured]),
            "total_seconds": distribution([run["total_seconds"] for run in measured]),
            "output_tokens": distribution([run["output_tokens"] for run in measured]),
            "output_tps": distribution([run["output_tps"] for run in measured]),
            "stream_event_interarrival_seconds": distribution([
                value
                for run in measured
                for value in run["stream_trace"]["event_interarrival_seconds"]
            ]),
            "client_token_itl_seconds": distribution([
                value
                for run in measured
                if run["stream_trace"]["client_token_itl_valid"]
                for value in run["stream_trace"]["client_token_itl_seconds"]
            ]),
            "client_token_itl_valid_runs": sum(
                run["stream_trace"]["client_token_itl_valid"] for run in measured
            ),
            "coalesced_stream_events": sum(
                run["stream_trace"]["coalesced_event_count"] for run in measured
            ),
            "unique_output_hashes": len({run["output_token_sha256"] for run in measured}),
        },
        "runs": runs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

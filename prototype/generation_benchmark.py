#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible completion server with an exact token prompt."""

from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import hashlib
import json
import math
import statistics
import struct
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
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


def pooled_output_tps(runs: list[dict]) -> float:
    """Compute aggregate steady-decode throughput without averaging ratios."""

    output_tokens = sum(max(0, int(run["output_tokens"]) - 1) for run in runs)
    stream_seconds = sum(float(run["stream_seconds"]) for run in runs)
    return output_tokens / stream_seconds if stream_seconds > 0 else 0.0


def routed_experts_from_event(event: dict) -> str | None:
    """Extract SGLang's routed-expert payload across response schema versions."""

    containers = [event.get("sgl_ext"), event.get("sglext")]
    containers.extend(
        choice.get("sgl_ext") or choice.get("sglext")
        for choice in event.get("choices") or []
        if isinstance(choice, dict)
    )
    for container in containers:
        if not isinstance(container, dict):
            continue
        value = container.get("routed_experts")
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            for item in value.values():
                if isinstance(item, str) and item:
                    return item
    return None


def decode_routed_experts(
    encoded: str,
    num_layers: int,
    top_k: int,
    experts_per_layer: int,
) -> list[list[list[int]]]:
    """Decode SGLang's flattened base64 int32 `[token, layer, top_k]` tensor."""

    if num_layers <= 0 or top_k <= 0 or experts_per_layer <= 0:
        raise ValueError("routed-expert dimensions must be positive")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("routed experts are not valid base64") from exc
    if len(raw) % 4:
        raise ValueError("routed-expert payload is not aligned to int32")
    count = len(raw) // 4
    token_stride = num_layers * top_k
    if count == 0 or count % token_stride:
        raise ValueError(
            f"routed-expert element count {count} is not divisible by "
            f"num_layers*top_k={token_stride}"
        )
    expert_ids = list(struct.unpack(f"<{count}i", raw))
    invalid = [expert for expert in expert_ids if not 0 <= expert < experts_per_layer]
    if invalid:
        raise ValueError(f"routed-expert payload contains invalid expert id {invalid[0]}")

    routes = []
    for token_start in range(0, count, token_stride):
        token_routes = []
        for layer_start in range(token_start, token_start + token_stride, top_k):
            selected = expert_ids[layer_start:layer_start + top_k]
            if len(set(selected)) != top_k:
                raise ValueError("a token-layer route contains duplicate expert ids")
            token_routes.append(selected)
        routes.append(token_routes)
    return routes


def expert_owner(expert_id: int, experts_per_layer: int, ep_size: int) -> int:
    if experts_per_layer % ep_size:
        raise ValueError("experts_per_layer must be divisible by ep_size")
    return expert_id // (experts_per_layer // ep_size)


def summarize_expert_routes(
    captures: list[dict],
    num_layers: int,
    top_k: int,
    experts_per_layer: int,
    ep_size: int,
) -> dict:
    """Build placement inputs: per-layer hotness, co-activation, and EP fanout."""

    experts_per_worker = experts_per_layer // ep_size
    if experts_per_worker * ep_size != experts_per_layer:
        raise ValueError("experts_per_layer must be divisible by ep_size")
    expert_counts = [[0] * experts_per_layer for _ in range(num_layers)]
    pair_counts = [Counter() for _ in range(num_layers)]
    worker_counts = [[0] * ep_size for _ in range(num_layers)]
    fanout_counts = Counter()
    token_layer_rows = 0
    request_summaries = []

    for capture in captures:
        routes = capture["routes"]
        request_summaries.append({
            "request_index": capture["request_index"],
            "server_request_id": capture.get("server_request_id"),
            "start_token_position": capture["start_token_position"],
            "captured_tokens": len(routes),
        })
        for token_routes in routes:
            if len(token_routes) != num_layers:
                raise ValueError("captured route has the wrong number of layers")
            for layer, experts in enumerate(token_routes):
                if len(experts) != top_k:
                    raise ValueError("captured route has the wrong top-k width")
                owners = set()
                for expert in experts:
                    expert_counts[layer][expert] += 1
                    owner = expert_owner(expert, experts_per_layer, ep_size)
                    worker_counts[layer][owner] += 1
                    owners.add(owner)
                for left, right in combinations(sorted(experts), 2):
                    pair_counts[layer][(left, right)] += 1
                fanout_counts[len(owners)] += 1
                token_layer_rows += 1

    layer_summaries = []
    for layer in range(num_layers):
        layer_summaries.append({
            "layer": layer,
            "expert_activation_counts": expert_counts[layer],
            "worker_activation_counts": worker_counts[layer],
            "coactivation_pairs": [
                {"experts": [left, right], "count": count}
                for (left, right), count in sorted(
                    pair_counts[layer].items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        })

    return {
        "schema": "dai-expert-placement-summary.v1",
        "request_count": len(captures),
        "token_layer_rows": token_layer_rows,
        "num_layers": num_layers,
        "top_k": top_k,
        "experts_per_layer": experts_per_layer,
        "expert_parallel_size": ep_size,
        "placement": [
            {
                "worker_rank": rank,
                "expert_id_start": rank * experts_per_worker,
                "expert_id_end_exclusive": (rank + 1) * experts_per_worker,
            }
            for rank in range(ep_size)
        ],
        "worker_fanout_token_layer_counts": {
            str(fanout): fanout_counts.get(fanout, 0)
            for fanout in range(1, ep_size + 1)
        },
        "cross_worker_token_layer_fraction": (
            sum(count for fanout, count in fanout_counts.items() if fanout > 1)
            / token_layer_rows
            if token_layer_rows
            else 0.0
        ),
        "requests": request_summaries,
        "layers": layer_summaries,
    }


def write_expert_route_jsonl(
    captures: list[dict],
    output: Path,
    prompt_tokens: int,
    experts_per_layer: int,
    ep_size: int,
) -> None:
    """Write one auditable row per request/token/layer, optionally gzip-compressed."""

    output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if output.suffix == ".gz" else open
    with opener(output, "wt", encoding="utf-8") as handle:
        for capture in captures:
            for token_offset, token_routes in enumerate(capture["routes"]):
                token_position = capture["start_token_position"] + token_offset
                for layer, experts in enumerate(token_routes):
                    handle.write(json.dumps({
                        "request_index": capture["request_index"],
                        "server_request_id": capture.get("server_request_id"),
                        "measured": capture["measured"],
                        "token_position": token_position,
                        "phase": "prompt" if token_position < prompt_tokens else "decode",
                        "layer": layer,
                        "expert_ids": experts,
                        "worker_ranks": [
                            expert_owner(expert, experts_per_layer, ep_size)
                            for expert in experts
                        ],
                    }, separators=(",", ":")) + "\n")


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
    return_routed_experts: bool = False,
    routed_experts_start_len: int = 0,
) -> dict:
    request_body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if return_routed_experts:
        request_body.update({
            "return_routed_experts": True,
            "routed_experts_start_len": routed_experts_start_len,
        })
    payload = json.dumps(request_body).encode("utf-8")
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
    server_request_id = None
    routed_experts_base64 = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if event.get("id"):
                server_request_id = event["id"]
            if event.get("usage"):
                usage = event["usage"]
            routed = routed_experts_from_event(event)
            if routed:
                if routed_experts_base64 and routed_experts_base64 != routed:
                    raise RuntimeError("server returned conflicting routed-expert payloads")
                routed_experts_base64 = routed
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
        "server_request_id": server_request_id,
        "routed_experts_base64": routed_experts_base64,
    }


def flush_server_cache(endpoint: str, timeout: float) -> None:
    """Flush SGLang's radix cache so repeated prompts measure cold prefill."""

    query = urllib.parse.urlencode({"timeout": timeout})
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/flush_cache?" + query,
        data=b"",
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout + 10) as response:
        response.read()


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
    parser.add_argument(
        "--cache-policy",
        choices=("warm", "cold"),
        default="warm",
        help="Use the radix cache or flush it before each request after the first warmup",
    )
    parser.add_argument("--nonce", default="dai-generation-v1")
    parser.add_argument(
        "--prompt-file",
        help="Use diverse UTF-8 source text instead of the synthetic repeated seed",
    )
    parser.add_argument(
        "--capture-routed-experts",
        action="store_true",
        help="Ask an instrumented SGLang server to return per-token/layer expert IDs",
    )
    parser.add_argument("--routed-experts-start-len", type=int, default=0)
    parser.add_argument("--routed-expert-layers", type=int, default=48)
    parser.add_argument("--routed-expert-top-k", type=int, default=8)
    parser.add_argument("--experts-per-layer", type=int, default=128)
    parser.add_argument("--expert-parallel-size", type=int, default=4)
    parser.add_argument(
        "--expert-routes-output",
        help="Write exact request/token/layer routes as JSONL or JSONL.GZ",
    )
    parser.add_argument(
        "--expert-summary-output",
        help="Write expert hotness, co-activation, ownership, and worker fanout JSON",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.cache_policy == "cold" and args.warmups < 1:
        parser.error("--cache-policy cold requires at least one warmup")
    if args.capture_routed_experts:
        if not args.expert_routes_output or not args.expert_summary_output:
            parser.error(
                "--capture-routed-experts requires --expert-routes-output and "
                "--expert-summary-output"
            )
        if args.routed_experts_start_len < 0:
            parser.error("--routed-experts-start-len must be non-negative")
        if args.experts_per_layer % args.expert_parallel_size:
            parser.error("--experts-per-layer must be divisible by --expert-parallel-size")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    source_text = (
        Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else None
    )
    prompt, prompt_ids = exact_token_prompt(
        tokenizer, args.prompt_tokens, args.nonce, source_text
    )
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    runs = []
    route_captures = []
    cache_flush_count = 0
    for index in range(args.warmups + args.repetitions):
        # SGLang can hang if /flush_cache is the first request after startup.
        # Let one unmeasured request initialize the runtime, then make every
        # subsequent request cold without including flush time in TTFT.
        if args.cache_policy == "cold" and index > 0:
            flush_server_cache(args.endpoint, min(args.timeout, 60.0))
            cache_flush_count += 1
        result = stream_completion(
            args.endpoint,
            args.model,
            prompt,
            args.max_tokens,
            args.timeout,
            return_routed_experts=args.capture_routed_experts,
            routed_experts_start_len=args.routed_experts_start_len,
        )
        trace = streaming_trace(result.pop("stream_events"), tokenizer)
        routed_experts_base64 = result.pop("routed_experts_base64")
        output_ids = tokenizer.encode(result.pop("text"), add_special_tokens=False)
        output_count = int(
            (result.get("usage") or {}).get("completion_tokens") or len(output_ids)
        )
        if output_count != args.max_tokens:
            raise RuntimeError(
                f"request {index} returned {output_count} completion tokens; "
                f"expected exactly {args.max_tokens}"
            )
        decode_seconds = result["stream_seconds"]
        output_tps = (
            max(0, output_count - 1) / decode_seconds
            if decode_seconds > 0 and output_count > 1
            else 0.0
        )
        run = {
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
        }
        if args.capture_routed_experts:
            if not routed_experts_base64:
                raise RuntimeError(
                    f"request {index} did not return routed experts; ensure the server "
                    "was launched with --enable-return-routed-experts"
                )
            routes = decode_routed_experts(
                routed_experts_base64,
                args.routed_expert_layers,
                args.routed_expert_top_k,
                args.experts_per_layer,
            )
            run["routed_experts"] = {
                "encoding": "base64-little-endian-int32",
                "logical_shape": [
                    len(routes),
                    args.routed_expert_layers,
                    args.routed_expert_top_k,
                ],
                "start_token_position": args.routed_experts_start_len,
                "data": routed_experts_base64,
            }
            route_captures.append({
                "request_index": index,
                "server_request_id": run.get("server_request_id"),
                "measured": run["measured"],
                "start_token_position": args.routed_experts_start_len,
                "routes": routes,
            })
        runs.append(run)

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
        "cache_policy": args.cache_policy,
        "cache_flush_count": cache_flush_count,
        "routed_expert_capture": {
            "enabled": args.capture_routed_experts,
            "start_token_position": (
                args.routed_experts_start_len if args.capture_routed_experts else None
            ),
            "num_layers": (
                args.routed_expert_layers if args.capture_routed_experts else None
            ),
            "top_k": args.routed_expert_top_k if args.capture_routed_experts else None,
            "experts_per_layer": (
                args.experts_per_layer if args.capture_routed_experts else None
            ),
            "expert_parallel_size": (
                args.expert_parallel_size if args.capture_routed_experts else None
            ),
        },
        "summary": {
            "ttft_seconds": distribution([run["ttft_seconds"] for run in measured]),
            "total_seconds": distribution([run["total_seconds"] for run in measured]),
            "output_tokens": distribution([run["output_tokens"] for run in measured]),
            "output_tps": distribution([run["output_tps"] for run in measured]),
            "pooled_output_tps": pooled_output_tps(measured),
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
    if args.capture_routed_experts:
        write_expert_route_jsonl(
            route_captures,
            Path(args.expert_routes_output),
            args.prompt_tokens,
            args.experts_per_layer,
            args.expert_parallel_size,
        )
        measured_captures = [capture for capture in route_captures if capture["measured"]]
        expert_summary = summarize_expert_routes(
            measured_captures,
            args.routed_expert_layers,
            args.routed_expert_top_k,
            args.experts_per_layer,
            args.expert_parallel_size,
        )
        expert_summary.update({
            "benchmark_output": str(output),
            "routes_output": args.expert_routes_output,
            "prompt_sha256": prompt_sha256,
            "cache_policy": args.cache_policy,
        })
        summary_output = Path(args.expert_summary_output)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(
            json.dumps(expert_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"output": str(output), **report["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

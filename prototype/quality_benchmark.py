#!/usr/bin/env python3
"""Run an objective answer-quality suite against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import string
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from generation_benchmark import distribution


def chat_completion(
    endpoint: str, model: str, prompt: str, max_tokens: int, timeout: float
) -> dict:
    """Apply Qwen's chat template in non-thinking mode and return final content."""

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read().decode("utf-8"))
    finished = time.perf_counter()
    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError("chat completion returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if not isinstance(content, str) or not content:
        raise RuntimeError(
            "non-thinking chat completion returned no answer content"
            + (" and unexpectedly returned reasoning content" if reasoning else "")
        )
    return {
        "text": content,
        "reasoning_content_present": bool(reasoning),
        "finish_reason": choices[0].get("finish_reason"),
        "usage": result.get("usage"),
        "total_seconds": finished - started,
    }


def normalize_answer(value: str) -> str:
    """Apply the common SQuAD-style normalization used for short answers."""

    value = unicodedata.normalize("NFKC", value).casefold()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def score_response(case: dict, response: str) -> dict:
    scorer = case["scorer"]
    scorer_type = scorer["type"]
    stripped = response.strip()

    if scorer_type == "literal":
        expected = str(scorer["expected"])
        return {"passed": stripped == expected, "extracted": stripped, "expected": expected}

    if scorer_type == "normalized_exact":
        expected_values = scorer["expected"]
        if isinstance(expected_values, str):
            expected_values = [expected_values]
        normalized = normalize_answer(stripped)
        normalized_expected = [normalize_answer(str(value)) for value in expected_values]
        return {
            "passed": normalized in normalized_expected,
            "extracted": normalized,
            "expected": normalized_expected,
        }

    if scorer_type == "choice":
        expected = str(scorer["expected"]).upper()
        patterns = (
            r"^\s*(?:answer\s*(?:is|:)\s*)?[\(\[]?([A-Za-z])(?:[\)\].:\s]|$)",
            r"\banswer\s*(?:is|:)\s*[\(\[]?([A-Za-z])(?:[\)\].:\s]|$)",
        )
        extracted = None
        for pattern in patterns:
            match = re.search(pattern, stripped, flags=re.IGNORECASE)
            if match:
                extracted = match.group(1).upper()
                break
        return {"passed": extracted == expected, "extracted": extracted, "expected": expected}

    if scorer_type == "numeric":
        expected = float(scorer["expected"])
        tolerance = float(scorer.get("tolerance", 0.0))
        matches = re.findall(
            r"[-+]?(?:\d*\.\d+|\d+(?:,\d{3})*)(?:[eE][-+]?\d+)?", stripped
        )
        extracted = float(matches[-1].replace(",", "")) if matches else None
        return {
            "passed": extracted is not None and abs(extracted - expected) <= tolerance,
            "extracted": extracted,
            "expected": expected,
            "tolerance": tolerance,
        }

    if scorer_type == "contains_all":
        expected = [str(value) for value in scorer["expected"]]
        haystack = stripped if scorer.get("case_sensitive", False) else stripped.casefold()
        needles = expected if scorer.get("case_sensitive", False) else [
            value.casefold() for value in expected
        ]
        return {
            "passed": all(value in haystack for value in needles),
            "extracted": stripped,
            "expected": expected,
        }

    raise ValueError(f"unsupported scorer type {scorer_type!r}")


def load_cases(path: Path) -> list[dict]:
    cases = []
    seen_ids = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        case = json.loads(raw_line)
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"line {line_number}: id must be a non-empty string")
        if case_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate id {case_id!r}")
        if not isinstance(case.get("prompt"), str) or not case["prompt"]:
            raise ValueError(f"line {line_number}: prompt must be a non-empty string")
        if not isinstance(case.get("scorer"), dict) or "type" not in case["scorer"]:
            raise ValueError(f"line {line_number}: scorer.type is required")
        seen_ids.add(case_id)
        cases.append(case)
    if not cases:
        raise ValueError("quality dataset contains no cases")
    return cases


def run_suite(
    endpoint: str,
    model: str,
    variant: str,
    dataset_path: Path,
    repetitions: int,
    default_max_tokens: int,
    timeout: float,
) -> dict:
    dataset_bytes = dataset_path.read_bytes()
    cases = load_cases(dataset_path)
    results = []
    for case in cases:
        attempts = []
        for _ in range(repetitions):
            completion = chat_completion(
                endpoint,
                model,
                case["prompt"],
                int(case.get("max_tokens", default_max_tokens)),
                timeout,
            )
            response = completion.pop("text")
            scored = score_response(case, response)
            attempts.append({"response": response, **completion, **scored})
        results.append({
            "id": case["id"],
            "category": case.get("category", "uncategorized"),
            "prompt_sha256": hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest(),
            "score": statistics.fmean(float(attempt["passed"]) for attempt in attempts),
            "attempts": attempts,
        })

    categories = {}
    for category in sorted({result["category"] for result in results}):
        category_scores = [result["score"] for result in results if result["category"] == category]
        categories[category] = {
            "cases": len(category_scores),
            "mean_score": statistics.fmean(category_scores),
        }
    attempts = [attempt for result in results for attempt in result["attempts"]]
    return {
        "schema": "dai-quality-benchmark.v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "model": model,
        "variant": variant,
        "dataset_name": dataset_path.name,
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "repetitions": repetitions,
        "summary": {
            "cases": len(results),
            "attempts": len(attempts),
            "mean_case_score": statistics.fmean(result["score"] for result in results),
            "categories": categories,
            "total_seconds": distribution([attempt["total_seconds"] for attempt in attempts]),
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--default-max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repetitions <= 0 or args.default_max_tokens <= 0:
        parser.error("repetitions and default-max-tokens must be positive")
    report = run_suite(
        args.endpoint,
        args.model,
        args.variant,
        Path(args.dataset),
        args.repetitions,
        args.default_max_tokens,
        args.timeout,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

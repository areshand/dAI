#!/usr/bin/env python3
"""Evaluate an acceptance-aware async draft/verify component model.

This is a screening model, not an end-to-end benchmark.  It deliberately labels
acceptance and verification values as assumptions until request-aligned traces
are available from the target runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def expected_tokens(alpha: float, gamma: int) -> float:
    if alpha == 1.0:
        return float(gamma)
    return (1.0 - alpha**gamma) / (1.0 - alpha)


def evaluate(
    *,
    draft_ms: float,
    rtt_ms: float,
    verify_ms: float,
    alpha: float,
    gamma: int,
    baseline_ms_per_token: float,
) -> dict[str, float | int | bool]:
    full_hit = alpha**gamma
    target_path_ms = rtt_ms + verify_ms
    async_cycle_ms = (
        full_hit * max(draft_ms, target_path_ms)
        + (1.0 - full_hit) * (draft_ms + target_path_ms)
    )
    tokens = expected_tokens(alpha, gamma)
    ms_per_token = async_cycle_ms / tokens
    return {
        "gamma": gamma,
        "alpha_assumption": alpha,
        "full_hit_probability": full_hit,
        "draft_ms": draft_ms,
        "rtt_ms": rtt_ms,
        "verify_ms_assumption": verify_ms,
        "expected_cycle_ms": async_cycle_ms,
        "expected_tokens_per_cycle": tokens,
        "expected_ms_per_token": ms_per_token,
        "expected_tokens_per_second": 1000.0 / ms_per_token,
        "speedup_vs_baseline": baseline_ms_per_token / ms_per_token,
        "beats_baseline": ms_per_token < baseline_ms_per_token,
        "meets_100ms_goal": ms_per_token < 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-profile", required=True, type=Path)
    parser.add_argument("--rtt-ms", required=True, type=float)
    parser.add_argument("--verify-ms", required=True, type=float)
    parser.add_argument("--alpha", required=True, type=float)
    parser.add_argument("--baseline-ms-per-token", type=float, default=16.589)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be in [0, 1]")
    if min(args.rtt_ms, args.verify_ms, args.baseline_ms_per_token) < 0:
        parser.error("timings must be non-negative")

    profile = json.loads(args.draft_profile.read_text())
    summary = profile.get("summary", profile)
    rows = []
    for gamma_text, stats in sorted(
        summary["round_seconds"].items(), key=lambda item: int(item[0])
    ):
        gamma = int(gamma_text)
        rows.append(
            evaluate(
                draft_ms=stats["mean"] * 1000.0,
                rtt_ms=args.rtt_ms,
                verify_ms=args.verify_ms,
                alpha=args.alpha,
                gamma=gamma,
                baseline_ms_per_token=args.baseline_ms_per_token,
            )
        )
    result = {
        "schema": "dai-async-overlap-screen.v1",
        "warning": (
            "Screening model only. Alpha and verification latency are assumptions "
            "unless supplied from request-aligned target traces."
        ),
        "draft_profile": str(args.draft_profile),
        "baseline_ms_per_token": args.baseline_ms_per_token,
        "cells": rows,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()

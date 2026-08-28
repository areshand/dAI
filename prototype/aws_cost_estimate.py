#!/usr/bin/env python3
"""Estimate current dAI experiment cost from live tagged AWS resources."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


# Linux On-Demand rates from the AWS Price List Bulk API on 2026-08-21.
# These are deliberately region-scoped; an unknown region/type is never guessed.
EC2_HOURLY_USD = {
    "us-west-2": {
        "c7i.xlarge": 0.1785,
        "r7i.4xlarge": 1.0584,
        "g5.12xlarge": 5.672,
    }
}

# gp3 includes 3,000 IOPS and 125 MiB/s. AWS bills additions per month.
GP3_MONTHLY_USD = {
    "us-west-2": {
        "gb": 0.08,
        "extra_iops": 0.005,
        "extra_mibps": 0.04,
    }
}

HOURS_PER_MONTH = 730.0
ACTIVE_COMPUTE_STATES = {"pending", "running", "stopping", "shutting-down"}


@dataclass
class RunEstimate:
    hourly_usd: float = 0.0
    accrued_usd: float = 0.0
    ttl_ceiling_usd: float = 0.0
    unknown_cost: bool = False


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def tag_value(resource: dict[str, Any], key: str, default: str = "-") -> str:
    for tag in resource.get("Tags", []):
        if tag.get("Key") == key:
            return str(tag.get("Value", default))
    return default


def elapsed_hours(start: datetime | None, now: datetime) -> float:
    if start is None:
        return 0.0
    return max(0.0, (now - start).total_seconds() / 3600.0)


def remaining_hours(expires_at: str | None, now: datetime) -> float:
    expiry = parse_time(expires_at)
    if expiry is None:
        return 0.0
    return max(0.0, (expiry - now).total_seconds() / 3600.0)


def load_ec2_rates(region: str) -> dict[str, float]:
    rates = dict(EC2_HOURLY_USD.get(region, {}))
    raw_overrides = os.environ.get("DAI_EC2_HOURLY_RATES_JSON", "{}")
    try:
        overrides = json.loads(raw_overrides)
        if not isinstance(overrides, dict):
            raise ValueError("expected a JSON object")
        rates.update({str(key): float(value) for key, value in overrides.items()})
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid DAI_EC2_HOURLY_RATES_JSON: {exc}") from exc
    return rates


def gp3_hourly_usd(volume: dict[str, Any], region: str) -> float | None:
    rates = GP3_MONTHLY_USD.get(region)
    if volume.get("VolumeType") != "gp3" or rates is None:
        return None
    size = float(volume.get("Size", 0))
    extra_iops = max(0.0, float(volume.get("Iops", 3000)) - 3000.0)
    extra_mibps = max(0.0, float(volume.get("Throughput", 125)) - 125.0)
    monthly = (
        size * rates["gb"]
        + extra_iops * rates["extra_iops"]
        + extra_mibps * rates["extra_mibps"]
    )
    return monthly / HOURS_PER_MONTH


def money(value: float | None, unknown: bool = False) -> str:
    if value is None or unknown:
        return "n/a"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in rows)
    return "\n".join(lines)


def estimate(
    instances: list[dict[str, Any]],
    volumes: list[dict[str, Any]],
    region: str,
    now: datetime,
) -> tuple[list[list[str]], list[list[str]], dict[str, RunEstimate], list[str]]:
    ec2_rates = load_ec2_rates(region)
    runs: dict[str, RunEstimate] = defaultdict(RunEstimate)
    instance_rows: list[list[str]] = []
    volume_rows: list[list[str]] = []
    warnings: list[str] = []

    for instance in sorted(instances, key=lambda item: (tag_value(item, "RunId"), item.get("InstanceId", ""))):
        run_id = tag_value(instance, "RunId", "untagged")
        state = instance.get("State", {}).get("Name", "unknown")
        instance_type = str(instance.get("InstanceType", "unknown"))
        rate = ec2_rates.get(instance_type)
        active = state in ACTIVE_COMPUTE_STATES
        hours = elapsed_hours(parse_time(instance.get("LaunchTime")), now)
        accrued = rate * hours if rate is not None and active else None
        hourly = rate if rate is not None and active else 0.0
        remaining = remaining_hours(tag_value(instance, "ExpiresAt", ""), now)
        unknown = rate is None or state == "stopped"

        runs[run_id].hourly_usd += hourly
        if accrued is not None:
            runs[run_id].accrued_usd += accrued
            runs[run_id].ttl_ceiling_usd += accrued + hourly * remaining
        if unknown:
            runs[run_id].unknown_cost = True

        if rate is None:
            warnings.append(
                f"No {region} EC2 rate for {instance_type}; add it with "
                f"DAI_EC2_HOURLY_RATES_JSON='{{\"{instance_type}\": RATE}}'."
            )
        if state == "stopped":
            warnings.append(
                f"{instance.get('InstanceId', 'unknown')} is stopped; its prior runtime cannot be derived from live state."
            )

        instance_rows.append(
            [
                run_id,
                str(instance.get("InstanceId", "-")),
                state,
                instance_type,
                str(instance.get("Placement", {}).get("AvailabilityZone", "-")),
                money(rate),
                f"{hours:.2f}h" if active else "-",
                money(accrued, unknown=unknown),
                tag_value(instance, "ExpiresAt"),
            ]
        )

    for volume in sorted(volumes, key=lambda item: (tag_value(item, "RunId"), item.get("VolumeId", ""))):
        run_id = tag_value(volume, "RunId", "untagged")
        rate = gp3_hourly_usd(volume, region)
        hours = elapsed_hours(parse_time(volume.get("CreateTime")), now)
        accrued = rate * hours if rate is not None else None
        remaining = remaining_hours(tag_value(volume, "ExpiresAt", ""), now)

        if rate is None:
            runs[run_id].unknown_cost = True
            warnings.append(
                f"No {region} EBS rate for volume type {volume.get('VolumeType', 'unknown')}."
            )
        else:
            runs[run_id].hourly_usd += rate
            runs[run_id].accrued_usd += accrued or 0.0
            runs[run_id].ttl_ceiling_usd += (accrued or 0.0) + rate * remaining

        volume_rows.append(
            [
                run_id,
                str(volume.get("VolumeId", "-")),
                str(volume.get("State", "unknown")),
                str(volume.get("VolumeType", "unknown")),
                f"{volume.get('Size', 0)} GiB",
                str(volume.get("Iops", "-")),
                str(volume.get("Throughput", "-")),
                money(rate),
                money(accrued),
            ]
        )

    return instance_rows, volume_rows, runs, list(dict.fromkeys(warnings))


def aws_json(profile: str, region: str, arguments: list[str]) -> dict[str, Any]:
    command = ["aws", *arguments, "--profile", profile, "--region", region, "--output", "json"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def fetch_resources(profile: str, region: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    instance_response = aws_json(
        profile,
        region,
        [
            "ec2",
            "describe-instances",
            "--filters",
            "Name=tag:Project,Values=dAI",
            "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down",
        ],
    )
    volume_response = aws_json(
        profile,
        region,
        ["ec2", "describe-volumes", "--filters", "Name=tag:Project,Values=dAI"],
    )
    instances = [
        instance
        for reservation in instance_response.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    return instances, list(volume_response.get("Volumes", []))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="mi:scratchpad")
    parser.add_argument("--region", default="us-west-2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        instances, volumes = fetch_resources(args.profile, args.region)
        instance_rows, volume_rows, runs, warnings = estimate(
            instances, volumes, args.region, datetime.now(timezone.utc)
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"dAI scratchpad resources in {args.region} (profile {args.profile})")
    if instance_rows:
        print("\nEC2 instances")
        print(
            format_table(
                ["RunId", "Instance", "State", "Type", "AZ", "Rate/h", "Elapsed", "Est. EC2", "ExpiresAt"],
                instance_rows,
            )
        )
    else:
        print("\nNo live or stopped dAI EC2 instances.")

    if volume_rows:
        print("\nEBS volumes")
        print(
            format_table(
                ["RunId", "Volume", "State", "Type", "Size", "IOPS", "MiB/s", "Rate/h", "Est. EBS"],
                volume_rows,
            )
        )
    else:
        print("No live dAI EBS volumes.")

    if runs:
        summary_rows = [
            [
                run_id,
                money(run.hourly_usd),
                money(run.accrued_usd, unknown=run.unknown_cost),
                money(run.ttl_ceiling_usd, unknown=run.unknown_cost),
            ]
            for run_id, run in sorted(runs.items())
        ]
        print("\nEstimated run cost (EC2 + live EBS)")
        print(format_table(["RunId", "Current burn", "Accrued", "TTL ceiling"], summary_rows))

    print("\nEstimate only: excludes S3, network transfer, taxes, credits, and account discounts.")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

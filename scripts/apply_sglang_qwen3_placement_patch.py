#!/usr/bin/env python3
"""Apply the pinned Qwen3 logical-to-physical expert dispatch fix in-place."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_V0516_SHA256 = "b18eb188c594c41ff58debe6df72cf852975b0504b5ae0513ccb4be75fea1bc2"
PATCH_MARKER = "dai_nontrivial_expert_location_fix"


def patch_text(source: str, expected_sha256: str | None = None) -> str:
    if PATCH_MARKER in source:
        return source
    actual_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"refusing to patch unexpected qwen3_moe.py: {actual_sha256}"
        )
    old_import = (
        "from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation\n"
    )
    new_import = (
        "from sglang.srt.eplb.expert_location import (\n"
        "    ModelConfigForExpertLocation,\n"
        "    get_global_expert_location_metadata,\n"
        ")\n"
    )
    old_topk = "        topk_output = self.topk(hidden_states, router_logits)\n"
    new_topk = (
        "        # dai_nontrivial_expert_location_fix: forward_normal loads remapped\n"
        "        # weights too, so it must translate logical router IDs to physical slots.\n"
        "        topk_output = self.topk(\n"
        "            hidden_states,\n"
        "            router_logits,\n"
        "            expert_location_dispatch_info=(\n"
        "                ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)\n"
        "                if get_global_expert_location_metadata() is not None\n"
        "                else None\n"
        "            ),\n"
        "        )\n"
    )
    if source.count(old_import) != 1 or source.count(old_topk) != 1:
        raise ValueError("pinned Qwen3 patch anchors are missing or ambiguous")
    return source.replace(old_import, new_import).replace(old_topk, new_topk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    source = args.path.read_text(encoding="utf-8")
    patched = patch_text(source, EXPECTED_V0516_SHA256)
    args.path.write_text(patched, encoding="utf-8")
    print(hashlib.sha256(patched.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    main()

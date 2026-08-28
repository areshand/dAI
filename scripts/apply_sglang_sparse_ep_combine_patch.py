#!/usr/bin/env python3
"""Apply the pinned experimental sparse EP combine patch in-place.

The prototype is deliberately narrow: for single-token DP-attention decode in
TP=DP=EP, ranks that own no selected physical expert skip the MoE output
combine. Active ranks send their partial output directly to the token owner.
Every other shape and topology falls back to SGLang's reduce_scatterv path.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_PLACEMENT_PATCHED_QWEN_SHA256 = (
    "43bc70dfdcdd80c1cc60fa1a2f1765e86ae0231373c6ab89ccb2995d037e84e0"
)
EXPECTED_V0516_COMMUNICATOR_SHA256 = (
    "2a16fcd432025a1fd6c382b91fdf2ed934bc396aebde4d7b56ad173fb4282a66"
)
QWEN_PATCH_MARKER = "dai_sparse_ep_record_active_ranks"
COMMUNICATOR_PATCH_MARKER = "dai_sparse_ep_combine"


def _check_hash(source: str, expected_sha256: str, label: str) -> None:
    actual_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"refusing to patch unexpected {label}: {actual_sha256}")


def patch_qwen_text(source: str) -> str:
    if QWEN_PATCH_MARKER in source:
        return source
    _check_hash(
        source, EXPECTED_PLACEMENT_PATCHED_QWEN_SHA256, "qwen3_moe.py"
    )

    old_import = "import logging\nimport math\n"
    new_import = "import logging\nimport math\nimport os\n"
    old_logger = "logger = logging.getLogger(__name__)\n_is_cuda = is_cuda()\n"
    new_logger = '''logger = logging.getLogger(__name__)

# dai_sparse_ep_record_active_ranks: transient metadata produced immediately
# before the layer communicator combines this MoE block's partial outputs.
_DAI_SPARSE_EP_ENABLED = os.environ.get("DAI_SPARSE_EP_COMBINE") == "1"
_DAI_SPARSE_EP_ACTIVE_RANKS = None
_DAI_SPARSE_EP_P2P_GROUP = None
_DAI_SPARSE_EP_METRICS = {
    "eligible_calls": 0,
    "sparse_calls": 0,
    "dense_fallback_calls": 0,
    "active_rank_histogram": {},
    "payload_bytes": 0,
}


def _dai_record_sparse_ep_active_ranks(topk_ids, experts_per_rank):
    global _DAI_SPARSE_EP_ACTIVE_RANKS
    if not _DAI_SPARSE_EP_ENABLED or topk_ids.shape[0] != 1:
        _DAI_SPARSE_EP_ACTIVE_RANKS = None
        return
    # This eight-ID device-to-host copy is intentionally part of the prototype
    # latency. A production implementation should keep routing metadata on GPU.
    physical_ids = topk_ids.detach().reshape(-1).to("cpu").tolist()
    _DAI_SPARSE_EP_ACTIVE_RANKS = tuple(
        sorted({int(expert_id) // experts_per_rank for expert_id in physical_ids})
    )


def _dai_maybe_log_sparse_ep_metrics():
    calls = _DAI_SPARSE_EP_METRICS["eligible_calls"]
    if calls == 1 or calls % 1536 == 0:
        logger.info("dai_sparse_ep_metrics=%s", _DAI_SPARSE_EP_METRICS)


def dai_sparse_ep_reduce_scatterv(global_hidden_states, hidden_states, sizes):
    """Sparse single-owner combine; return False to use the stock collective."""
    global _DAI_SPARSE_EP_ACTIVE_RANKS, _DAI_SPARSE_EP_P2P_GROUP
    active_ranks = _DAI_SPARSE_EP_ACTIVE_RANKS
    _DAI_SPARSE_EP_ACTIVE_RANKS = None
    if not _DAI_SPARSE_EP_ENABLED or active_ranks is None:
        return False

    from sglang.srt.distributed import get_tp_group

    group = get_tp_group()
    if (
        group.world_size != len(sizes)
        or sum(sizes) != 1
        or global_hidden_states.shape[0] != 1
        or len(active_ranks) == 0
        or any(rank < 0 or rank >= group.world_size for rank in active_ranks)
    ):
        return False

    owner_ranks = [rank for rank, size in enumerate(sizes) if size == 1]
    if len(owner_ranks) != 1:
        return False

    owner_rank = owner_ranks[0]
    active_count = len(active_ranks)
    _DAI_SPARSE_EP_METRICS["eligible_calls"] += 1
    histogram = _DAI_SPARSE_EP_METRICS["active_rank_histogram"]
    histogram[active_count] = histogram.get(active_count, 0) + 1
    if active_count == group.world_size:
        _DAI_SPARSE_EP_METRICS["dense_fallback_calls"] += 1
        _dai_maybe_log_sparse_ep_metrics()
        return False

    # A distinct communicator is required: inactive ranks may enqueue the next
    # TP collective before the owner finishes receiving this layer's outputs.
    # Reusing the TP communicator would violate NCCL operation ordering.
    if _DAI_SPARSE_EP_P2P_GROUP is None:
        if torch.distributed.get_world_size() != group.world_size:
            return False
        _DAI_SPARSE_EP_P2P_GROUP = torch.distributed.new_group(
            ranks=group.ranks, backend="nccl"
        )

    local_rank = group.rank_in_group
    if local_rank == owner_rank:
        hidden_states.zero_()
        if owner_rank in active_ranks:
            hidden_states.copy_(global_hidden_states)
        for source_rank in active_ranks:
            if source_rank != owner_rank:
                received = torch.empty_like(global_hidden_states)
                torch.distributed.recv(
                    received,
                    src=group.ranks[source_rank],
                    group=_DAI_SPARSE_EP_P2P_GROUP,
                )
                hidden_states.add_(received)
    elif local_rank in active_ranks:
        torch.distributed.send(
            global_hidden_states.contiguous(),
            dst=group.ranks[owner_rank],
            group=_DAI_SPARSE_EP_P2P_GROUP,
        )

    _DAI_SPARSE_EP_METRICS["sparse_calls"] += 1
    remote_senders = active_count - int(owner_rank in active_ranks)
    _DAI_SPARSE_EP_METRICS["payload_bytes"] += (
        remote_senders * global_hidden_states.numel() * global_hidden_states.element_size()
    )
    _dai_maybe_log_sparse_ep_metrics()
    return True


_is_cuda = is_cuda()
'''
    old_init = "        self.ep_size = get_parallel().moe_ep_size\n        self.layer_id = layer_id\n"
    new_init = (
        "        self.ep_size = get_parallel().moe_ep_size\n"
        "        self.layer_id = layer_id\n"
        "        self.dai_experts_per_rank = config.num_experts // self.ep_size\n"
    )
    old_topk_tail = (
        "            ),\n"
        "        )\n"
        "        final_hidden_states = self.experts(hidden_states, topk_output)\n"
    )
    new_topk_tail = (
        "            ),\n"
        "        )\n"
        "        _dai_record_sparse_ep_active_ranks(\n"
        "            topk_output.topk_ids, self.dai_experts_per_rank\n"
        "        )\n"
        "        final_hidden_states = self.experts(hidden_states, topk_output)\n"
    )
    anchors = (old_import, old_logger, old_init, old_topk_tail)
    if any(source.count(anchor) != 1 for anchor in anchors):
        raise ValueError("pinned Qwen3 sparse-combine anchors are missing or ambiguous")
    return (
        source.replace(old_import, new_import)
        .replace(old_logger, new_logger)
        .replace(old_init, new_init)
        .replace(old_topk_tail, new_topk_tail)
    )


def patch_communicator_text(source: str) -> str:
    if COMMUNICATOR_PATCH_MARKER in source:
        return source
    _check_hash(source, EXPECTED_V0516_COMMUNICATOR_SHA256, "communicator.py")
    old_block = '''        if should_use_dp_reduce_scatterv():
            get_tp_group().reduce_scatterv(
                global_hidden_states,
                output=hidden_states,
                sizes=get_dp_global_num_tokens(),
            )
'''
    new_block = '''        if should_use_dp_reduce_scatterv():
            sizes = get_dp_global_num_tokens()
            # dai_sparse_ep_combine: only the explicitly enabled, single-token
            # EP prototype may bypass the stock full-rank collective.
            from sglang.srt.models.qwen3_moe import (
                dai_sparse_ep_reduce_scatterv,
            )

            if not dai_sparse_ep_reduce_scatterv(
                global_hidden_states, hidden_states, sizes
            ):
                get_tp_group().reduce_scatterv(
                    global_hidden_states,
                    output=hidden_states,
                    sizes=sizes,
                )
'''
    if source.count(old_block) != 1:
        raise ValueError("pinned communicator sparse-combine anchor is missing")
    return source.replace(old_block, new_block)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qwen_path", type=Path)
    parser.add_argument("communicator_path", type=Path)
    args = parser.parse_args()

    qwen_source = args.qwen_path.read_text(encoding="utf-8")
    communicator_source = args.communicator_path.read_text(encoding="utf-8")
    patched_qwen = patch_qwen_text(qwen_source)
    patched_communicator = patch_communicator_text(communicator_source)
    args.qwen_path.write_text(patched_qwen, encoding="utf-8")
    args.communicator_path.write_text(patched_communicator, encoding="utf-8")
    print(
        "qwen3_moe.py="
        + hashlib.sha256(patched_qwen.encode("utf-8")).hexdigest()
        + " communicator.py="
        + hashlib.sha256(patched_communicator.encode("utf-8")).hexdigest()
    )


if __name__ == "__main__":
    main()

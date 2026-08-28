import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("apply_sglang_sparse_ep_combine_patch.py")
SPEC = importlib.util.spec_from_file_location("apply_sparse_ep_patch", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ApplySparseEpPatchTest(unittest.TestCase):
    def test_qwen_patch_records_physical_expert_ranks(self):
        source = """import logging
import math
logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
        self.ep_size = get_parallel().moe_ep_size
        self.layer_id = layer_id
        # dai_nontrivial_expert_location_fix
        topk_output = self.topk(
            hidden_states,
            router_logits,
            expert_location_dispatch_info=(
                None
            ),
        )
        final_hidden_states = self.experts(hidden_states, topk_output)
"""
        original = MODULE.EXPECTED_PLACEMENT_PATCHED_QWEN_SHA256
        try:
            MODULE.EXPECTED_PLACEMENT_PATCHED_QWEN_SHA256 = __import__(
                "hashlib"
            ).sha256(source.encode()).hexdigest()
            patched = MODULE.patch_qwen_text(source)
        finally:
            MODULE.EXPECTED_PLACEMENT_PATCHED_QWEN_SHA256 = original
        self.assertIn(MODULE.QWEN_PATCH_MARKER, patched)
        self.assertIn("topk_output.topk_ids", patched)
        self.assertIn("dai_sparse_ep_reduce_scatterv", patched)
        self.assertEqual(MODULE.patch_qwen_text(patched), patched)

    def test_communicator_patch_retains_collective_fallback(self):
        source = """        if should_use_dp_reduce_scatterv():
            get_tp_group().reduce_scatterv(
                global_hidden_states,
                output=hidden_states,
                sizes=get_dp_global_num_tokens(),
            )
"""
        original = MODULE.EXPECTED_V0516_COMMUNICATOR_SHA256
        try:
            MODULE.EXPECTED_V0516_COMMUNICATOR_SHA256 = __import__(
                "hashlib"
            ).sha256(source.encode()).hexdigest()
            patched = MODULE.patch_communicator_text(source)
        finally:
            MODULE.EXPECTED_V0516_COMMUNICATOR_SHA256 = original
        self.assertIn(MODULE.COMMUNICATOR_PATCH_MARKER, patched)
        self.assertIn("if not dai_sparse_ep_reduce_scatterv", patched)
        self.assertIn("get_tp_group().reduce_scatterv", patched)
        self.assertEqual(MODULE.patch_communicator_text(patched), patched)

    def test_rejects_unexpected_sources(self):
        with self.assertRaisesRegex(ValueError, "unexpected qwen3_moe.py"):
            MODULE.patch_qwen_text("wrong")
        with self.assertRaisesRegex(ValueError, "unexpected communicator.py"):
            MODULE.patch_communicator_text("wrong")


if __name__ == "__main__":
    unittest.main()

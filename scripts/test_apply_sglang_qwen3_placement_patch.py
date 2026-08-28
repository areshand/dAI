import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("apply_sglang_qwen3_placement_patch.py")
SPEC = importlib.util.spec_from_file_location("apply_sglang_patch", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ApplySglangPatchTest(unittest.TestCase):
    def test_adds_dispatch_info_to_normal_path(self):
        source = (
            "from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation\n"
            "def forward_normal():\n"
            "        topk_output = self.topk(hidden_states, router_logits)\n"
        )
        patched = MODULE.patch_text(source)
        self.assertIn(MODULE.PATCH_MARKER, patched)
        self.assertIn("get_global_expert_location_metadata", patched)
        self.assertIn("ExpertLocationDispatchInfo.init_new", patched)
        self.assertEqual(MODULE.patch_text(patched), patched)

    def test_rejects_unexpected_pinned_source(self):
        with self.assertRaisesRegex(ValueError, "unexpected qwen3_moe.py"):
            MODULE.patch_text("wrong", "expected")


if __name__ == "__main__":
    unittest.main()

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_low_vram_ep.py")
SPEC = importlib.util.spec_from_file_location("analyze_low_vram_ep", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AnalyzeLowVramEpTest(unittest.TestCase):
    def test_network_bytes_excludes_loopback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.json"
            path.write_text(json.dumps([
                {"ifname": "lo", "stats64": {"rx": {"bytes": 100}, "tx": {"bytes": 100}}},
                {"ifname": "ens5", "stats64": {"rx": {"bytes": 300}, "tx": {"bytes": 400}}},
            ]))
            self.assertEqual(MODULE.network_bytes(path), 700)

    def test_measured_hashes_ignore_warmups(self):
        report = {"runs": [
            {"measured": False, "output_token_sha256": "warmup"},
            {"measured": True, "output_token_sha256": "answer"},
            {"measured": True, "output_token_sha256": "answer"},
        ]}
        self.assertEqual(MODULE.measured_hashes(report), ["answer"])


if __name__ == "__main__":
    unittest.main()

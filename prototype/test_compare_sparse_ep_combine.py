import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("compare_sparse_ep_combine.py")
SPEC = importlib.util.spec_from_file_location("compare_sparse_ep", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CompareSparseEpCombineTest(unittest.TestCase):
    def test_network_bytes_excludes_loopback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.json"
            path.write_text(
                json.dumps(
                    [
                        {"ifname": "lo", "stats64": {"rx": {"bytes": 100}, "tx": {"bytes": 100}}},
                        {"ifname": "ens5", "stats64": {"rx": {"bytes": 200}, "tx": {"bytes": 300}}},
                    ]
                )
            )
            self.assertEqual(MODULE.network_bytes(path), 500)

    def test_extracts_last_sparse_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(
                "x dai_sparse_ep_metrics={'sparse_calls': 1, 'active_rank_histogram': {3: 1}}\n"
                "x dai_sparse_ep_metrics={'sparse_calls': 2, 'active_rank_histogram': {2: 1, 3: 1}}\n"
            )
            self.assertEqual(MODULE.last_sparse_metrics(path)["sparse_calls"], 2)

    def test_percent_change(self):
        self.assertAlmostEqual(MODULE.percent_change(11, 10), 10)


if __name__ == "__main__":
    unittest.main()

import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("optimize_expert_placement.py")
SPEC = importlib.util.spec_from_file_location("optimize_expert_placement", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OptimizeExpertPlacementTest(unittest.TestCase):
    def test_capacity_balance_and_locality(self):
        with tempfile.TemporaryDirectory() as directory:
            routes = Path(directory) / "routes.jsonl.gz"
            rows = []
            # Under the trivial 0..3 / 4..7 placement, every hot pair crosses.
            hot_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
            for request_index in range(4):
                for repetition in range(10):
                    for layer in range(2):
                        pair = hot_pairs[repetition % len(hot_pairs)]
                        rows.append({
                            "request_index": request_index,
                            "measured": True,
                            "layer": layer,
                            "expert_ids": pair,
                        })
            with gzip.open(routes, "wt", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")

            config, report = MODULE.optimize(
                routes, num_layers=2, experts=8, workers=2,
                train_requests=2, max_load_ratio=1.05,
            )

            for mapping in config["physical_to_logical_map"]:
                self.assertEqual(sorted(mapping), list(range(8)))
                self.assertEqual(len(mapping[:4]), 4)
                self.assertEqual(len(mapping[4:]), 4)
            self.assertLess(
                report["optimized"]["held_out"]["cross_worker_expert_pair_fraction"],
                report["baseline"]["held_out"]["cross_worker_expert_pair_fraction"],
            )
            self.assertLessEqual(
                report["optimized"]["train"]["worst_layer_max_to_mean_activation_ratio"],
                1.05,
            )

    def test_requires_held_out_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            routes = Path(directory) / "routes.jsonl.gz"
            with gzip.open(routes, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "request_index": 0,
                    "measured": True,
                    "layer": 0,
                    "expert_ids": [0, 1],
                }) + "\n")
            with self.assertRaisesRegex(ValueError, "both training and held-out"):
                MODULE.optimize(
                    routes, num_layers=1, experts=2, workers=2,
                    train_requests=1, max_load_ratio=1.05,
                )


if __name__ == "__main__":
    unittest.main()

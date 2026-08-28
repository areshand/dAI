import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("compare_expert_placement.py")
SPEC = importlib.util.spec_from_file_location("compare_expert_placement", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def benchmark(tps, ttft=0.1, prompt_tokens=1000):
    return {
        "prompt_sha256": "prompt",
        "prompt_tokens": prompt_tokens,
        "max_tokens": 256,
        "repetitions": 10,
        "cache_policy": "cold",
        "sampling": {"temperature": 0, "seed": 1234},
        "runs": [
            {
                "measured": True,
                "output_tokens": 256,
                "output_token_ids": list(range(256)),
                "output_token_sha256": "hash",
            }
            for _ in range(10)
        ],
        "summary": {
            "pooled_output_tps": tps,
            "ttft_seconds": {"mean": ttft},
            "total_seconds": {"mean": 20},
            "unique_output_hashes": 1,
        },
    }


class CompareExpertPlacementTest(unittest.TestCase):
    def test_bracketed_speedup_and_drift(self):
        optimization = {
            "baseline": {"held_out": {"mean_workers_per_token_layer": 3.6}},
            "optimized": {"held_out": {"mean_workers_per_token_layer": 3.0}},
            "runtime_boundary": "collectives remain four-rank",
        }
        result = MODULE.compare(benchmark(10), benchmark(15), benchmark(10.4), optimization)
        self.assertAlmostEqual(result["optimized_speedup_vs_bracketed_trivial"], 15 / 10.2)
        self.assertTrue(result["baseline_drift_within_5_percent"])
        self.assertTrue(result["output_hash_sets_match"])

    def test_rejects_workload_mismatch(self):
        optimization = {
            "baseline": {"held_out": {}},
            "optimized": {"held_out": {}},
            "runtime_boundary": "boundary",
        }
        with self.assertRaisesRegex(ValueError, "prompt_tokens"):
            MODULE.compare(benchmark(10), benchmark(15, prompt_tokens=999), benchmark(10), optimization)

    def test_rejects_degenerate_or_incomplete_retokenized_output(self):
        optimization = {
            "baseline": {"held_out": {}},
            "optimized": {"held_out": {}},
            "runtime_boundary": "boundary",
        }
        invalid = benchmark(15)
        for row in invalid["runs"]:
            row["output_token_ids"] = [90440] * 128
        with self.assertRaisesRegex(ValueError, "complete measured requests"):
            MODULE.compare(benchmark(10), invalid, benchmark(10), optimization)


if __name__ == "__main__":
    unittest.main()

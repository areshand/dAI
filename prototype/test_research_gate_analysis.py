import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from research_gate_analysis import analyze, bootstrap_mean_difference, parse_machine_counts


def report(variant, periods_ms, output_hash="same", valid_trace=True):
    runs = []
    for index, period in enumerate(periods_ms):
        itl = period / 1000.0
        runs.append({
            "index": index,
            "measured": True,
            "output_tokens": 3,
            "stream_seconds": 2 * itl,
            "output_token_sha256": output_hash,
            "stream_trace": {
                "client_token_itl_valid": valid_trace,
                "client_token_itl_seconds": [itl, itl] if valid_trace else [],
            },
        })
    return {"variant": variant, "runs": runs}


class ResearchGateAnalysisTest(unittest.TestCase):
    def test_machine_count_parser(self):
        self.assertEqual(parse_machine_counts(["remote=1", "swarm=4"]), {"remote": 1, "swarm": 4})

    def test_bootstrap_detects_clear_improvement(self):
        result = bootstrap_mean_difference([10, 10, 10], [7, 7, 7], 1000, 1)
        self.assertEqual(result["estimate"], -3)
        self.assertEqual(result["ci95_high"], -3)

    def test_analysis_requires_correctness_sla_and_added_machine(self):
        result = analyze(
            [report("baseline", [20, 20, 20]), report("remote", [10, 10, 10])],
            {"remote": 1}, 0.10, 0.05, 100.0, 1000, 1,
        )
        remote = result["cells"][1]
        self.assertTrue(remote["gates"]["minimum_improvement"])
        self.assertTrue(remote["gates"]["client_itl_sla"])
        self.assertTrue(remote["gates"]["machine_causal_improvement"])

    def test_invalid_stream_trace_blocks_sla_claim(self):
        result = analyze(
            [report("baseline", [20, 20]), report("remote", [10, 10], valid_trace=False)],
            {"remote": 1}, 0.10, 0.05, 100.0, 1000, 1,
        )
        self.assertFalse(result["cells"][1]["gates"]["client_itl_sla"])
        self.assertFalse(result["cells"][1]["gates"]["machine_causal_improvement"])


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from compare_quality_results import analyze, paired_bootstrap_delta


def quality_report(variant, scores):
    return {
        "variant": variant,
        "dataset_name": "suite.jsonl",
        "dataset_sha256": "abc",
        "cases": [{"id": f"case-{index:03d}", "score": score} for index, score in enumerate(scores)],
    }


def generation_report(variant, tps=120.0, event_p99=0.02, ttft_p99=0.10):
    return {
        "variant": variant,
        "summary": {
            "output_tps": {"mean": tps},
            "stream_event_interarrival_seconds": {"p99": event_p99},
            "ttft_seconds": {"p99": ttft_p99},
        },
    }


class QualityComparisonTest(unittest.TestCase):
    def test_paired_bootstrap_identical(self):
        result = paired_bootstrap_delta([1.0] * 10, [1.0] * 10, 100, 1)
        self.assertEqual(result["estimate"], 0.0)
        self.assertEqual(result["ci95_low"], 0.0)

    def test_combined_gate_passes_with_sufficient_identical_suite(self):
        scores = [1.0] * 100
        result = analyze(
            [quality_report("baseline", scores), quality_report("candidate", scores)],
            [generation_report("baseline", 60.0), generation_report("candidate")],
            margin=0.02,
            min_cases=100,
            target_tps=100.0,
            max_event_gap_ms=100.0,
            max_ttft_ms=250.0,
            bootstrap_samples=100,
            seed=1,
        )
        candidate = result["cells"][1]
        self.assertTrue(candidate["gates"]["quality_noninferior"])
        self.assertTrue(candidate["gates"]["quality_qualified_100_tps"])

    def test_smoke_suite_cannot_qualify(self):
        scores = [1.0] * 16
        result = analyze(
            [quality_report("baseline", scores), quality_report("candidate", scores)],
            [generation_report("baseline"), generation_report("candidate")],
            margin=0.02,
            min_cases=100,
            target_tps=100.0,
            max_event_gap_ms=100.0,
            max_ttft_ms=250.0,
            bootstrap_samples=100,
            seed=1,
        )
        self.assertFalse(result["cells"][1]["gates"]["sample_sufficient"])
        self.assertFalse(result["cells"][1]["gates"]["quality_qualified_100_tps"])

    def test_latency_tail_blocks_combined_gate(self):
        scores = [1.0] * 100
        result = analyze(
            [quality_report("baseline", scores), quality_report("candidate", scores)],
            [generation_report("baseline"), generation_report("candidate", event_p99=0.2)],
            margin=0.02,
            min_cases=100,
            target_tps=100.0,
            max_event_gap_ms=100.0,
            max_ttft_ms=250.0,
            bootstrap_samples=100,
            seed=1,
        )
        self.assertFalse(result["cells"][1]["gates"]["stream_pause_sla"])
        self.assertFalse(result["cells"][1]["gates"]["quality_qualified_100_tps"])


if __name__ == "__main__":
    unittest.main()

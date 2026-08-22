import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from generation_benchmark import distribution, percentile, streaming_trace


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


class StatisticsTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.50), 2.0)
        self.assertAlmostEqual(percentile([1.0, 2.0], 0.95), 1.95)

    def test_distribution(self):
        result = distribution([1.0, 2.0, 3.0])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["p50"], 2.0)
        self.assertAlmostEqual(result["p99"], 2.98)

    def test_streaming_trace_accepts_one_token_per_event(self):
        trace = streaming_trace([
            {"arrival_seconds": 0.1, "text": "a"},
            {"arrival_seconds": 0.3, "text": "b"},
            {"arrival_seconds": 0.6, "text": "c"},
        ], CharacterTokenizer())
        self.assertTrue(trace["client_token_itl_valid"])
        self.assertAlmostEqual(trace["client_token_itl_seconds"][0], 0.2)
        self.assertAlmostEqual(trace["client_token_itl_seconds"][1], 0.3)
        self.assertEqual(trace["coalesced_event_count"], 0)

    def test_streaming_trace_flags_coalesced_events(self):
        trace = streaming_trace([
            {"arrival_seconds": 0.1, "text": "ab"},
            {"arrival_seconds": 0.2, "text": "c"},
        ], CharacterTokenizer())
        self.assertFalse(trace["client_token_itl_valid"])
        self.assertEqual(trace["coalesced_event_count"], 1)
        self.assertEqual(trace["client_token_itl_seconds"], [])
        self.assertEqual(trace["event_interarrival_seconds"], [0.1])


if __name__ == "__main__":
    unittest.main()

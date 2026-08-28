import base64
import gzip
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import generation_benchmark
from generation_benchmark import (
    decode_routed_experts,
    distribution,
    expert_owner,
    flush_server_cache,
    percentile,
    pooled_output_tps,
    routed_experts_from_event,
    summarize_expert_routes,
    streaming_trace,
    write_expert_route_jsonl,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b'{"success":true}'


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

    def test_pooled_output_tps_uses_total_tokens_over_total_time(self):
        runs = [
            {"output_tokens": 11, "stream_seconds": 1.0},
            {"output_tokens": 11, "stream_seconds": 3.0},
        ]
        self.assertEqual(pooled_output_tps(runs), 5.0)

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

    def test_flush_server_cache_posts_with_timeout(self):
        with patch.object(
            generation_benchmark.urllib.request,
            "urlopen",
            return_value=FakeResponse(),
        ) as opened:
            flush_server_cache("http://server/", 30.0)

        request = opened.call_args.args[0]
        self.assertEqual(
            request.full_url, "http://server/flush_cache?timeout=30.0"
        )
        self.assertEqual(request.method, "POST")
        self.assertEqual(opened.call_args.kwargs["timeout"], 40.0)

    def test_routed_experts_from_response_level_extension(self):
        self.assertEqual(
            routed_experts_from_event({
                "choices": [],
                "sgl_ext": {"routed_experts": "encoded"},
            }),
            "encoded",
        )

    def test_routed_experts_from_choice_extension(self):
        self.assertEqual(
            routed_experts_from_event({
                "choices": [{"sglext": {"routed_experts": "encoded"}}],
            }),
            "encoded",
        )

    def test_decode_routed_experts_uses_token_layer_topk_shape(self):
        expert_ids = list(range(8))
        encoded = base64.b64encode(struct.pack("<8i", *expert_ids)).decode()
        self.assertEqual(
            decode_routed_experts(encoded, num_layers=2, top_k=2, experts_per_layer=8),
            [
                [[0, 1], [2, 3]],
                [[4, 5], [6, 7]],
            ],
        )

    def test_decode_routed_experts_rejects_invalid_shape(self):
        encoded = base64.b64encode(struct.pack("<3i", 0, 1, 2)).decode()
        with self.assertRaisesRegex(ValueError, "not divisible"):
            decode_routed_experts(encoded, num_layers=2, top_k=2, experts_per_layer=8)

    def test_expert_owner_uses_contiguous_trivial_placement(self):
        self.assertEqual(expert_owner(0, 128, 4), 0)
        self.assertEqual(expert_owner(31, 128, 4), 0)
        self.assertEqual(expert_owner(32, 128, 4), 1)
        self.assertEqual(expert_owner(127, 128, 4), 3)

    def test_summary_counts_hotness_pairs_and_worker_fanout(self):
        captures = [{
            "request_index": 1,
            "server_request_id": "cmpl-test",
            "measured": True,
            "start_token_position": 10,
            "routes": [
                [[0, 4], [1, 2]],
                [[0, 1], [5, 7]],
            ],
        }]
        result = summarize_expert_routes(
            captures,
            num_layers=2,
            top_k=2,
            experts_per_layer=8,
            ep_size=2,
        )
        self.assertEqual(result["token_layer_rows"], 4)
        self.assertEqual(result["worker_fanout_token_layer_counts"], {"1": 3, "2": 1})
        self.assertEqual(result["cross_worker_token_layer_fraction"], 0.25)
        self.assertEqual(result["layers"][0]["expert_activation_counts"][0], 2)
        self.assertEqual(
            result["layers"][0]["coactivation_pairs"],
            [
                {"experts": [0, 1], "count": 1},
                {"experts": [0, 4], "count": 1},
            ],
        )

    def test_route_jsonl_identifies_request_token_layer_and_workers(self):
        captures = [{
            "request_index": 3,
            "server_request_id": "cmpl-test",
            "measured": True,
            "start_token_position": 9,
            "routes": [[[0, 4], [1, 2]]],
        }]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "routes.jsonl.gz"
            write_expert_route_jsonl(
                captures,
                output,
                prompt_tokens=10,
                experts_per_layer=8,
                ep_size=2,
            )
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["request_index"], 3)
        self.assertEqual(rows[0]["token_position"], 9)
        self.assertEqual(rows[0]["phase"], "prompt")
        self.assertEqual(rows[0]["expert_ids"], [0, 4])
        self.assertEqual(rows[0]["worker_ranks"], [0, 1])


if __name__ == "__main__":
    unittest.main()

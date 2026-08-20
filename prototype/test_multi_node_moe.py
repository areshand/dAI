import unittest

from multi_node_moe import build_placements, parse_worker


class MultiNodeMoeTest(unittest.TestCase):
    def test_parse_worker(self):
        self.assertEqual(parse_worker("near=10.0.0.1:50123"), ("near", "10.0.0.1", 50123))

    def test_latency_aware_places_hot_experts_on_faster_worker(self):
        discovery = {
            "near": {"probe_ms": {"p50": 0.5}},
            "far": {"probe_ms": {"p50": 1.5}},
        }
        placements = build_placements(["near", "far"], discovery, 7)
        self.assertEqual(placements["latency_aware"][0], "near")
        self.assertEqual(placements["latency_aware"][1], "near")
        self.assertEqual(sorted(placements["random"].values()), ["far", "far", "near", "near"])
        self.assertNotEqual(placements["random"], placements["hot_near"])
        self.assertNotEqual(placements["random"], placements["hot_far"])

    def test_requires_preregistered_worker_names(self):
        discovery = {"a": {"probe_ms": {"p50": 1}}, "b": {"probe_ms": {"p50": 2}}}
        with self.assertRaises(ValueError):
            build_placements(["a", "b"], discovery, 7)


if __name__ == "__main__":
    unittest.main()

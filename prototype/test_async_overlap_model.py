import importlib.util
import pathlib
import unittest


PATH = pathlib.Path(__file__).with_name("async_overlap_model.py")
SPEC = importlib.util.spec_from_file_location("async_overlap_model", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class AsyncOverlapModelTests(unittest.TestCase):
    def test_expected_tokens_limits(self):
        self.assertEqual(MODULE.expected_tokens(0.0, 8), 1.0)
        self.assertEqual(MODULE.expected_tokens(1.0, 8), 8.0)

    def test_perfect_acceptance_divides_cycle_by_gamma(self):
        result = MODULE.evaluate(
            draft_ms=80.0,
            rtt_ms=5.0,
            verify_ms=20.0,
            alpha=1.0,
            gamma=4,
            baseline_ms_per_token=30.0,
        )
        self.assertEqual(result["expected_cycle_ms"], 80.0)
        self.assertEqual(result["expected_ms_per_token"], 20.0)
        self.assertTrue(result["beats_baseline"])

    def test_rejection_pays_serial_cycle(self):
        result = MODULE.evaluate(
            draft_ms=40.0,
            rtt_ms=5.0,
            verify_ms=20.0,
            alpha=0.0,
            gamma=4,
            baseline_ms_per_token=30.0,
        )
        self.assertEqual(result["expected_cycle_ms"], 65.0)
        self.assertEqual(result["expected_ms_per_token"], 65.0)


if __name__ == "__main__":
    unittest.main()

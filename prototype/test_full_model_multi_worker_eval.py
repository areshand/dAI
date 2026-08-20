import argparse
import unittest

from full_model_multi_worker_eval import distribution, parse_worker, percentile


class FullModelMultiWorkerHelpersTest(unittest.TestCase):
    def test_parse_worker(self):
        self.assertEqual(parse_worker("near=10.0.0.2:50126"), ("near", "10.0.0.2", 50126))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_worker("other=10.0.0.2:50126")

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.5), 2.5)

    def test_distribution(self):
        result = distribution([1.0, 2.0, 3.0])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["mean"], 2.0)
        self.assertEqual(result["p50"], 2.0)


if __name__ == "__main__":
    unittest.main()

import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import prepare_gsm8k_quality_suite as gsm8k


class Gsm8kPreparationTest(unittest.TestCase):
    def test_extract_expected(self):
        self.assertEqual(gsm8k.extract_expected("work\n#### 1,234.5"), 1234.5)

    def test_convert_is_deterministic_and_provenanced(self):
        source = b"\n".join([
            json.dumps({"question": "one", "answer": "work\n#### 1"}).encode(),
            json.dumps({"question": "two", "answer": "work\n#### 2"}).encode(),
            json.dumps({"question": "three", "answer": "work\n#### 3"}).encode(),
        ]) + b"\n"
        with patch.object(gsm8k, "GSM8K_SHA256", hashlib.sha256(source).hexdigest()):
            first = gsm8k.convert(source, 2, 7)
            second = gsm8k.convert(source, 2, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0]["source"]["dataset"], "GSM8K")
        self.assertEqual(first[0]["scorer"]["type"], "numeric")

    def test_hash_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "source hash mismatch"):
            gsm8k.convert(b"not the official source", 1, 1)


if __name__ == "__main__":
    unittest.main()

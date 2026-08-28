import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from quality_benchmark import chat_completion, load_cases, normalize_answer, score_response


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps({
            "choices": [{
                "message": {"content": "B", "reasoning_content": None},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 1},
        }).encode("utf-8")


class QualityScoringTest(unittest.TestCase):
    def test_chat_completion_disables_thinking(self):
        with patch("quality_benchmark.urllib.request.urlopen", return_value=FakeResponse()) as opened:
            result = chat_completion("http://server", "model", "prompt", 8, 30.0)
        request = opened.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://server/v1/chat/completions")
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(result["text"], "B")

    def test_normalized_answer(self):
        self.assertEqual(normalize_answer(" The, Blue  Car! "), "blue car")

    def test_literal_is_strict(self):
        case = {"scorer": {"type": "literal", "expected": "H2O"}}
        self.assertTrue(score_response(case, "H2O\n")["passed"])
        self.assertFalse(score_response(case, "h2o")["passed"])

    def test_choice_accepts_answer_prefix(self):
        case = {"scorer": {"type": "choice", "expected": "B"}}
        self.assertTrue(score_response(case, "Answer: B")["passed"])
        self.assertFalse(score_response(case, "A")["passed"])

    def test_numeric_uses_final_number_and_tolerance(self):
        case = {"scorer": {"type": "numeric", "expected": 3.14, "tolerance": 0.01}}
        self.assertTrue(score_response(case, "After 2 steps, the answer is 3.141")["passed"])

    def test_contains_all(self):
        case = {"scorer": {"type": "contains_all", "expected": ["Latency", "Throughput"]}}
        self.assertTrue(score_response(case, "throughput trades off with LATENCY")["passed"])

    def test_loader_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            row = {"id": "same", "prompt": "p", "scorer": {"type": "literal", "expected": "x"}}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()

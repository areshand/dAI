import importlib.util
import json
import pathlib
import sys
import unittest
from unittest.mock import patch


PATH = pathlib.Path(__file__).with_name("spec_logprob_probe.py")
sys.path.insert(0, str(PATH.parent))
SPEC = importlib.util.spec_from_file_location("spec_logprob_probe", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b'{"text":"ok","meta_info":{}}'


class SpecLogprobProbeTests(unittest.TestCase):
    def test_requests_greedy_target_logprobs(self):
        with patch.object(
            MODULE.urllib.request, "urlopen", return_value=FakeResponse()
        ) as opened:
            response = MODULE.generate("http://server/", "prompt", 17, 8)

        request = opened.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(response["text"], "ok")
        self.assertEqual(request.full_url, "http://server/generate")
        self.assertEqual(payload["sampling_params"]["temperature"], 0)
        self.assertEqual(payload["sampling_params"]["max_new_tokens"], 17)
        self.assertTrue(payload["return_logprob"])
        self.assertEqual(payload["top_logprobs_num"], 8)


if __name__ == "__main__":
    unittest.main()

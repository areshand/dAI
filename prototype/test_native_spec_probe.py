import importlib.util
import pathlib
import sys
import unittest


PATH = pathlib.Path(__file__).with_name("native_spec_probe.py")
sys.path.insert(0, str(PATH.parent))
SPEC = importlib.util.spec_from_file_location("native_spec_probe", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class NativeSpecProbeTests(unittest.TestCase):
    def test_extracts_only_spec_fields(self):
        details = MODULE.extract_spec_details(
            {
                "meta_info": {
                    "spec_accept_rate": 0.5,
                    "spec_verify_ct": 7,
                    "completion_tokens": 20,
                }
            }
        )
        self.assertEqual(details, {"spec_accept_rate": 0.5, "spec_verify_ct": 7})

    def test_missing_meta_is_empty(self):
        self.assertEqual(MODULE.extract_spec_details({}), {})


if __name__ == "__main__":
    unittest.main()

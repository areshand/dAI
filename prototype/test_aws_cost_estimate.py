import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from prototype.aws_cost_estimate import estimate, gp3_hourly_usd, load_ec2_rates


NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


class AwsCostEstimateTests(unittest.TestCase):
    def test_generation_run_includes_ec2_and_provisioned_gp3(self):
        tags = [
            {"Key": "Project", "Value": "dAI"},
            {"Key": "RunId", "Value": "dai-gen-test"},
            {"Key": "ExpiresAt", "Value": "2026-08-21T14:30:00Z"},
        ]
        instances = [
            {
                "InstanceId": "i-test",
                "InstanceType": "g5.12xlarge",
                "LaunchTime": "2026-08-21T11:30:00Z",
                "State": {"Name": "running"},
                "Placement": {"AvailabilityZone": "us-west-2a"},
                "Tags": tags,
            }
        ]
        volumes = [
            {
                "VolumeId": "vol-test",
                "VolumeType": "gp3",
                "Size": 180,
                "Iops": 8000,
                "Throughput": 1000,
                "CreateTime": "2026-08-21T11:30:00Z",
                "State": "in-use",
                "Tags": tags,
            }
        ]

        _, _, runs, warnings = estimate(instances, volumes, "us-west-2", NOW)
        expected_ebs_hourly = 74.4 / 730.0
        run = runs["dai-gen-test"]

        self.assertAlmostEqual(run.hourly_usd, 5.672 + expected_ebs_hourly)
        self.assertAlmostEqual(run.accrued_usd, (5.672 + expected_ebs_hourly) * 0.5)
        self.assertAlmostEqual(run.ttl_ceiling_usd, (5.672 + expected_ebs_hourly) * 3.0)
        self.assertFalse(run.unknown_cost)
        self.assertEqual(warnings, [])

    def test_gp3_included_performance_is_not_charged_twice(self):
        volume = {"VolumeType": "gp3", "Size": 16, "Iops": 3000, "Throughput": 125}
        self.assertAlmostEqual(gp3_hourly_usd(volume, "us-west-2"), 1.28 / 730.0)

    def test_unknown_type_requires_override(self):
        with patch.dict(os.environ, {"DAI_EC2_HOURLY_RATES_JSON": '{"custom.large": 1.25}'}, clear=False):
            rates = load_ec2_rates("us-west-2")
        self.assertEqual(rates["custom.large"], 1.25)

    def test_no_resources_has_no_runs(self):
        instance_rows, volume_rows, runs, warnings = estimate([], [], "us-west-2", NOW)
        self.assertEqual((instance_rows, volume_rows, dict(runs), warnings), ([], [], {}, []))


if __name__ == "__main__":
    unittest.main()

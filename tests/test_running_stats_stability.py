import numpy as np
import torch
import unittest

from lerobot.scripts.lerobot_data_stats import RunningStats as LeRobotRunningStats
from util_scripts.compute_norm_stats_multi import RunningStats as MultiRunningStats
from util_scripts.compute_norm_stats_single import RunningStats as SingleRunningStats


class RunningStatsStabilityTest(unittest.TestCase):
    def test_small_variance_around_large_offset_is_preserved(self):
        values = -3.3114 + 0.0006 * torch.sin(torch.arange(5000, dtype=torch.float32))
        expected = values.numpy().astype(np.float64)

        implementations = [
            ("single", SingleRunningStats),
            ("multi", MultiRunningStats),
            ("lerobot", LeRobotRunningStats),
        ]
        for name, running_stats_cls in implementations:
            with self.subTest(implementation=name):
                stats = running_stats_cls()
                for batch in values.chunk(50):
                    stats.update(batch)

                result = stats.get_statistics()

                self.assertEqual(result["count"], [5000])
                self.assertAlmostEqual(result["std"][0], expected.std(), delta=1e-12)
                self.assertAlmostEqual(result["mean"][0], expected.mean(), delta=1e-12)
                self.assertAlmostEqual(result["min"][0], expected.min(), delta=1e-12)
                self.assertAlmostEqual(result["max"][0], expected.max(), delta=1e-12)

    def test_multi_worker_payload_merge_preserves_small_variance(self):
        values = -3.3114 + 0.0006 * torch.sin(torch.arange(5000, dtype=torch.float32))
        expected = values.numpy().astype(np.float64)

        left = MultiRunningStats()
        right = MultiRunningStats()
        left.update(values[:2500])
        right.update(values[2500:])

        merged = MultiRunningStats.from_payload(left.to_payload())
        merged.merge(MultiRunningStats.from_payload(right.to_payload()))
        result = merged.get_statistics()

        self.assertEqual(result["count"], [5000])
        self.assertAlmostEqual(result["std"][0], expected.std(), delta=1e-12)
        self.assertAlmostEqual(result["mean"][0], expected.mean(), delta=1e-12)


if __name__ == "__main__":
    unittest.main()

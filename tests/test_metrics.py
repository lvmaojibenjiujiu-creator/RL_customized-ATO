from __future__ import annotations

import unittest

from rl_ato.env import throughput_normalized_inventory_ratio


class MetricTests(unittest.TestCase):
    def test_residual_inventory_ratio_uses_same_horizon_aggregation(self) -> None:
        end_inventories = [10.0, 20.0]
        replenishments = [5.0, 10.0]

        corrected = throughput_normalized_inventory_ratio(
            sum(end_inventories),
            sum(replenishments),
        )
        old_inconsistent = (sum(end_inventories) / 2.0) / sum(replenishments)

        self.assertAlmostEqual(corrected, 2.0)
        self.assertAlmostEqual(old_inconsistent, 1.0)


if __name__ == "__main__":
    unittest.main()

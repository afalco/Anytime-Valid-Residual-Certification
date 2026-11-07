"""Contract checks for the schedule diagnostic, not production QB tests."""

import math
import unittest

from run_schedule_comparison import METHODS, analytic_metrics, risk_level


class ScheduleChecks(unittest.TestCase):
    def test_budget(self):
        for method in METHODS[1:]:
            total = sum(risk_level(method, j, 0.05, 80) for j in range(1, 81))
            self.assertLessEqual(total, 0.05 + 1e-15)
        self.assertAlmostEqual(80 * risk_level('uniform', 1, 0.05, 80), 0.05)

    def test_no_global_dominance(self):
        self.assertLess(risk_level('uniform', 1, 0.05, 80), risk_level('power', 1, 0.05, 80))
        self.assertGreater(risk_level('uniform', 80, 0.05, 80), risk_level('power', 80, 0.05, 80))
        crossing = math.sqrt(6 * 80 / math.pi**2)
        self.assertEqual(math.floor(crossing) + 1, 7)

    def test_analytic_safety_and_counterexample(self):
        for method in METHODS[1:]:
            result = analytic_metrics(method, 80, 3, 0.05, 0.7)
            self.assertLessEqual(result['false_stop_rate'], 0.05)
            self.assertLessEqual(result['miscoverage_rate'], 0.05)
            self.assertGreaterEqual(result['mean_rank'], 1)
            self.assertLessEqual(result['mean_rank'], 80 + 1e-10)
        naive = analytic_metrics('naive', 80, 3, 0.05, 0.7)
        self.assertGreater(naive['false_stop_rate'], 0.05)
        self.assertGreater(naive['miscoverage_rate'], 0.05)


if __name__ == '__main__':
    unittest.main()

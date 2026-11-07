"""Frozen experimental-design and finite-horizon schedule checks."""

import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / 'revision_R1_study'))
from run_production import clean, expected, HERE
from certification import inspection_risk


class ProductionDesignTests(unittest.TestCase):
    def test_sensitivity_and_coverage_counts(self):
        protocol = json.loads((HERE / 'protocol.json').read_text())
        self.assertEqual(expected(protocol, 'coverage'), 4000)
        self.assertEqual(expected(protocol, 'sensitivity'), 3600)
        self.assertEqual(expected(protocol, 'graphs'), 90)

    def test_schedule_budget_and_out_of_horizon(self):
        for schedule in ('power', 'uniform'):
            self.assertLessEqual(sum(inspection_risk(0.05, j, 20, schedule) for j in range(1, 21)), 0.05 + 1e-15)
        with self.assertRaises(ValueError):
            inspection_risk(0.05, 21, 20, 'uniform')

    def test_nonfinite_checkpoint_values_are_explicit(self):
        encoded = clean(dict(upper=math.inf, conditional=math.nan))
        self.assertEqual(encoded, dict(upper='Infinity', conditional='NA'))
        json.dumps(encoded, allow_nan=False)


if __name__ == '__main__':
    unittest.main()

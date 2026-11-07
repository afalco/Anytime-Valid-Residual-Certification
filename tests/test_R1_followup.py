"""Implicit-operator algebra and access-cost contracts."""

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/revision_followup_study'))
from run_followup import SchurOperator, implicit_trial


class FollowupTests(unittest.TestCase):
    def test_schur_forward_and_transpose(self):
        a = SchurOperator(12, 123, 1e-5)
        dense = 1e-5 * np.eye(12) - a.c.toarray() @ np.linalg.solve(a.h.toarray(), a.d.toarray())
        rng = np.random.default_rng(42)
        x, y = rng.normal(size=(12, 3)), rng.normal(size=(12, 2))
        np.testing.assert_allclose(a @ x, dense @ x, atol=1e-14)
        np.testing.assert_allclose(a.T @ y, dense.T @ y, atol=1e-14)
        np.testing.assert_allclose(y.T @ (a @ x), (a.T @ y).T @ x, atol=1e-14)
        self.assertEqual(a.forward_rhs, 6)
        self.assertEqual(a.transpose_rhs, 4)

    def test_absolute_tolerance_and_paid_norm(self):
        p = dict(n=24, problem_seed=123, shift=1e-5, trial_seed=456, batch_size=7,
                 max_rank=10, block_size=5, alpha=.05, cert_size=80, epsilon=1.)
        energy = implicit_trial(p, 0, 'energy_scan')
        random = implicit_trial(p, 0, 'anytime_power')
        self.assertEqual(energy['norm_A'], 24)
        self.assertEqual(random['norm_A'], 0)
        self.assertEqual(energy['solver_A'], 24 + 5)
        self.assertEqual(random['solver_A'], 80 + 5)
        for row in (energy, random):
            self.assertEqual(row['solver_AT'], row['rank'])
            self.assertEqual(row['diagnostic_A'], 24)
            self.assertEqual(row['diagnostic_AT'], 0)
            self.assertEqual(row['certified'], 1)
            self.assertEqual(row['false_stop'], 0)
        self.assertEqual(energy['prefix_checksums'][0], random['prefix_checksums'][0])


if __name__ == '__main__':
    unittest.main()

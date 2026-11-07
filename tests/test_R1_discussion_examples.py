"""Deterministic checks of the examples in the R1 scientific discussion."""

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "real_matrix_study"))
from generate_benchmark_matrices import elliptic_laplacian_matrix


class DiscussionExamplesTests(unittest.TestCase):
    def test_rademacher_moments_and_zero_measurement(self):
        residual = np.array([[1., 2., -1.], [0., 1., 3.]])
        gram = residual.T @ residual
        signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
        measured = np.sum((signs @ residual.T) ** 2, axis=1)
        self.assertAlmostEqual(measured.mean(), np.trace(gram))
        off_diagonal = gram - np.diag(np.diag(gram))
        self.assertAlmostEqual(measured.var(), 2 * np.sum(off_diagonal ** 2))
        two_signs = np.array(list(itertools.product((-1., 1.), repeat=2)))
        self.assertEqual(np.mean(np.sum(two_signs, axis=1) ** 2 == 0), 0.5)

    def test_laplacian_is_full_rank_kronecker_sum(self):
        q = 17
        t = 2 * np.eye(q) - np.eye(q, k=1) - np.eye(q, k=-1)
        separated = np.kron(np.eye(q), t) + np.kron(t, np.eye(q))
        np.testing.assert_array_equal(elliptic_laplacian_matrix(q).toarray(), separated)
        eigenvalues = 2 - 2 * np.cos(np.arange(1, q + 1) * np.pi / (q + 1))
        analytic = np.sort((eigenvalues[:, None] + eigenvalues[None, :]).ravel())
        self.assertTrue(np.all(analytic > 0))
        np.testing.assert_allclose(np.linalg.eigvalsh(separated), analytic, atol=1e-13)

    def test_spd_residual_solution_error_bound(self):
        operator = np.array([[2., -1.], [-1., 2.]])
        solution = np.array([2., -3.])
        approximate = np.array([1., 0.])
        residual = operator @ (solution - approximate)
        gamma = np.linalg.eigvalsh(operator).min()
        self.assertLessEqual(np.linalg.norm(solution - approximate), np.linalg.norm(residual) / gamma)


if __name__ == "__main__":
    unittest.main()

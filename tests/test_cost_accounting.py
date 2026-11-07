"""Independent cost identities, coefficient caching, and paired-path checks."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / 'src'))
sys.path.insert(0, str(ROOT / 'experiments' / 'real_matrix_study'))
from cost_accounting import CostLedger, CoefficientCache, direct_stored_residual, energy_residual
import run_real_matrices as real


def run(a=None, **kwargs):
    a = np.eye(8) if a is None else a
    args = dict(rng=np.random.default_rng(1), basis_rng=np.random.default_rng(2), a=a,
                info=real.MatrixInfo('cost', *a.shape, int(a.nnz) if sp.issparse(a) else int(np.count_nonzero(a)), int(sp.issparse(a))),
                method='anytime', trial=0, block_size=3, cert_size=200, alpha=0.05,
                rel_tol=0, max_rank=5, rho_mode='conservative', cert_batch_size=17,
                rho_pilot_size=2, rho_pilot_shrink=0.5)
    args.update(kwargs)
    return real.run_one(**args)


class LedgerTests(unittest.TestCase):
    def test_vectors_and_blocks_count_right_hand_sides(self):
        ledger = CostLedger()
        a = np.ones((7, 5))
        ledger.apply(a, np.ones(5), 'qb')
        ledger.apply(a, np.ones((5, 3)), 'certification')
        ledger.apply(a, np.ones((7, 2)), 'b', transpose=True)
        self.assertEqual(ledger.total('A'), 4)
        self.assertEqual(ledger.total('AT'), 2)
        self.assertEqual(ledger.total('AT', diagnostic=True), 0)

    def test_timers_may_not_overlap(self):
        ledger = CostLedger()
        with ledger.measure('qb'):
            with self.assertRaises(ValueError):
                with ledger.measure('b'):
                    pass

    def test_coefficients_increment_without_recounting_old_columns(self):
        rng = np.random.default_rng(5)
        a = rng.standard_normal((9, 6))
        q, _ = np.linalg.qr(rng.standard_normal((9, 5)))
        ledger, cache = CostLedger(), CoefficientCache(6)
        for k in (2, 4, 5, 5):
            cache.update(a, q[:, :k], ledger)
        self.assertEqual(ledger.total('AT'), 5)
        self.assertTrue(np.allclose(cache.bt, a.T @ q))
        changed = q.copy()
        changed[:, [0, 1]] = changed[:, [1, 0]]
        with self.assertRaises(ValueError):
            cache.update(a, changed, ledger)

    def test_energy_identity_and_direct_diagnostic(self):
        rng = np.random.default_rng(8)
        a = rng.standard_normal((9, 6))
        q, _ = np.linalg.qr(rng.standard_normal((9, 3)))
        bt = a.T @ q
        actual = np.linalg.norm(a - q @ bt.T, 'fro')
        energy, clipped = energy_residual(np.linalg.norm(a, 'fro'), bt)
        self.assertFalse(clipped)
        self.assertAlmostEqual(actual, energy)
        self.assertAlmostEqual(actual, direct_stored_residual(a, q, bt, 2))
        self.assertAlmostEqual(actual, direct_stored_residual(sp.coo_matrix(a), q, bt, 2))

    def test_negative_energy_is_flagged_not_silently_treated_as_exact_zero(self):
        residual, clipped = energy_residual(1.0, np.array([[2.0]]))
        self.assertEqual(residual, 0.0)
        self.assertTrue(clipped)


class RunnerCostTests(unittest.TestCase):
    def test_final_b_and_cached_diagnostic_are_not_double_counted(self):
        row = run()
        counts = json.loads(row.costs_json)['counts']
        self.assertEqual(row.solver_A, 5 + 2 * 200)
        self.assertEqual(row.solver_AT, 5)
        self.assertEqual(counts['b']['AT'], row.rank)
        self.assertEqual(row.diagnostic_AT, 0)
        self.assertEqual(row.diagnostic_A, 0)
        self.assertEqual(row.residual_eval_matvecs, 0)
        self.assertEqual(row.norm_entries_read, 64)
        self.assertGreaterEqual(row.runtime_sec, row.solver_runtime_sec)

    def test_invalid_certification_has_no_operator_applications(self):
        row = run(cert_size=3)
        self.assertEqual(row.solver_A, 5)
        self.assertEqual(row.solver_AT, 5)
        self.assertEqual(json.loads(row.costs_json)['counts']['certification']['A'], 0)

    def test_pilot_directions_are_separate(self):
        row = run(rho_mode='pilot')
        counts = json.loads(row.costs_json)['counts']
        self.assertEqual(counts['pilot'], {'A': 4, 'AT': 4})
        self.assertEqual(row.solver_AT, 9)
        self.assertEqual(row.solver_A, 409)
        self.assertEqual(row.certificate_kind, 'gaussian_pilot_heuristic')

    def test_energy_baseline_has_no_random_certification_cost(self):
        row = run(method='energy', coefficient_mode='incremental', diagnostic_mode='direct')
        counts = json.loads(row.costs_json)['counts']
        self.assertEqual(row.solver_A, 5)
        self.assertEqual(row.solver_AT, 5)
        self.assertEqual(counts['certification'], {'A': 0, 'AT': 0})
        self.assertEqual(row.s_used, (0, 0))
        self.assertEqual(row.diagnostic_entries_materialized, 64)

    def test_incremental_and_final_b_have_same_count_and_residual(self):
        final, incremental = run(), run(coefficient_mode='incremental')
        self.assertEqual(final.solver_AT, incremental.solver_AT)
        self.assertEqual(final.rank, incremental.rank)
        self.assertAlmostEqual(final.residual_ratio, incremental.residual_ratio)

    def test_method_seeds_share_q_prefix_without_counterfactual_cost(self):
        original = real.qb_step
        traces = []
        for method, mode, tolerance in (('energy', 'incremental', 2), ('anytime', 'final', 0)):
            trace = []
            def observe(*args, **kwargs):
                q = original(*args, **kwargs)
                trace.append(q.copy())
                return q
            with patch.object(real, 'qb_step', side_effect=observe):
                row = run(method=method, coefficient_mode=mode, rel_tol=tolerance)
            traces.append(trace)
            self.assertEqual(json.loads(row.costs_json)['counts']['qb']['A'], row.rank)
        self.assertTrue(np.array_equal(traces[0][0], traces[1][0]))
        self.assertEqual(len(traces[0]), 1)
        self.assertEqual(len(traces[1]), 2)

    def test_rectangular_sparse_final_and_incremental(self):
        a = sp.csr_matrix(np.eye(9, 6))
        for mode in ('final', 'incremental'):
            row = run(a, coefficient_mode=mode, diagnostic_mode='direct')
            self.assertEqual(row.solver_AT, row.rank)
            self.assertEqual(row.norm_entries_read, a.nnz)

    def test_oracle_materialization_is_recorded_not_called_free_matrix_free(self):
        row = run(rho_mode='oracle')
        self.assertEqual(row.oracle_entries_materialized, row.inspections * 64)
        self.assertGreater(row.oracle_runtime_sec, 0)


if __name__ == '__main__':
    unittest.main()

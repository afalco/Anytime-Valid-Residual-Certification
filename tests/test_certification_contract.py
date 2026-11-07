"""R1 output-state, positivity, and finite-inspection contract tests."""

import dataclasses
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
for directory in ('src', 'synthetic_qb_study', 'real_matrix_study'):
    sys.path.insert(0, str(ROOT / 'experiments' / directory))

from certification import CertificateResult, gaussian_certificate, inflation_factor, sketch_size
import run_synthetic_qb as synthetic
import run_real_matrices as real
import run_adaptive_qb_coverage as coverage


def synthetic_run(a=None, **overrides):
    args = dict(rng=np.random.default_rng(42), a=np.eye(8) if a is None else a,
                spectrum='test', method='anytime', trial=0, block_size=3,
                cert_size=3, alpha=0.05, rel_tol=0.01, max_rank=5, rho_mode='conservative')
    args.update(overrides)
    return synthetic.run_one(**args)


def real_run(a=None, **overrides):
    a = np.eye(8) if a is None else a
    args = dict(rng=np.random.default_rng(42), a=a,
                info=real.MatrixInfo('test', *a.shape, int(a.nnz) if sp.issparse(a) else int(np.count_nonzero(a)), int(sp.issparse(a))),
                method='anytime', trial=0, block_size=3, cert_size=3,
                alpha=0.05, rel_tol=0.01, max_rank=5, rho_mode='conservative',
                cert_batch_size=2, rho_pilot_size=2, rho_pilot_shrink=0.5)
    args.update(overrides)
    return real.run_one(**args)


class PositivityTests(unittest.TestCase):
    def test_invalid_denominator_uses_no_randomness_or_products(self):
        for s in (3, 4):
            rng, product = Mock(), Mock()
            out = gaussian_certificate(rng, product, 8, math.exp(-1), s, 1)
            self.assertTrue(math.isinf(out.upper))
            self.assertFalse(out.denominator_valid)
            self.assertEqual(out.probes_used, 0)
            rng.standard_normal.assert_not_called()
            product.assert_not_called()

    def test_positive_denominator_and_batching(self):
        out = gaussian_certificate(np.random.default_rng(1), lambda x: x, 8, math.exp(-1), 5, 1, 2)
        self.assertTrue(out.denominator_valid)
        self.assertEqual(out.probes_used, 5)
        self.assertTrue(math.isfinite(out.upper))

    def test_known_zero_precedes_positivity_guard(self):
        rng = Mock()
        out = gaussian_certificate(rng, Mock(), 8, 0.01, 1, 1, known_zero=True)
        self.assertEqual(out.upper, 0)
        self.assertEqual(out.probes_used, 0)
        rng.standard_normal.assert_not_called()

    def test_adaptive_size_guarantees_positivity(self):
        for alpha in (0.1, 0.001, 1e-20):
            for rho in (1, 10, 100):
                size = sketch_size(alpha, 3, rho, 16)
                self.assertLessEqual(inflation_factor(alpha, size, rho), 0.5 + 1e-15)

    def test_invalid_parameters(self):
        for alpha, size, rho in ((0, 5, 1), (1, 5, 1), (0.1, 0, 1), (0.1, 2.5, 1), (0.1, 5, math.inf), (0.1, 5, 0)):
            with self.assertRaises(ValueError):
                inflation_factor(alpha, size, rho)
        with self.assertRaises(ValueError):
            sketch_size(0.1, 3, 1, 4)


class RunnerContractTests(unittest.TestCase):
    def test_cap_above_tolerance_is_not_false_stop(self):
        for runner in (synthetic_run, real_run):
            row = runner()
            self.assertEqual(row.status, 'cap_reached')
            self.assertFalse(row.certified)
            self.assertFalse(row.tolerance_met)
            self.assertEqual(row.false_stop, 0)
            self.assertEqual(row.n_invalid_denominator, 2)
            self.assertEqual(row.s_used, (0, 0))
            self.assertTrue(row.final_certificate_infinite)

    def test_below_tolerance_need_not_be_certified(self):
        for runner in (synthetic_run, real_run):
            row = runner(rel_tol=2)
            self.assertTrue(row.tolerance_met)
            self.assertFalse(row.certified)
            self.assertEqual(row.false_stop, 0)

    def test_certified_failure_and_last_inspection_acceptance(self):
        for runner, module in ((synthetic_run, synthetic), (real_run, real)):
            outcomes = [CertificateResult(math.inf, False, 0), CertificateResult(0.0, True, 3)]
            with patch.object(module, 'residual_certificate', side_effect=outcomes):
                row = runner()
            self.assertEqual(row.inspections, 2)
            self.assertEqual(row.status, 'certified')
            self.assertTrue(row.certified)
            self.assertFalse(row.tolerance_met)
            self.assertEqual(row.false_stop, 1)

    def test_zero_input_does_not_require_effective_rank(self):
        for runner, module, name in ((synthetic_run, synthetic, 'effective_rank'), (real_run, real, 'effective_rank_oracle')):
            with patch.object(module, name, side_effect=AssertionError('must not evaluate rank at known zero')):
                row = runner(np.zeros((7, 5)), rho_mode='oracle')
            self.assertEqual(row.rank, 0)
            value = row.certificate if isinstance(row, synthetic.TrialResult) else row.certificate_ratio
            self.assertEqual(value, 0)
            self.assertEqual(row.status, 'certified')
            self.assertTrue(row.tolerance_met)
            self.assertEqual(row.s_used, (0,))

    def test_sparse_zero(self):
        row = real_run(sp.csr_matrix((7, 5)))
        self.assertEqual(row.status, 'certified')
        self.assertEqual(row.rank, 0)

    def test_blocks_truncate_to_cap_and_dimensions(self):
        for runner, module in ((synthetic_run, synthetic), (real_run, real)):
            with patch.object(module, 'qb_step', wraps=module.qb_step) as step:
                row = runner(np.eye(7, 5), max_rank=4)
            self.assertEqual(row.rank, 4)
            self.assertEqual([call.args[-1] for call in step.call_args_list], [3, 1])
            row = runner(np.eye(7, 5), max_rank=100)
            self.assertLessEqual(row.rank, 5)

    def test_rank_deficient_blocks_do_not_invent_rank(self):
        a = np.outer(np.arange(1., 9), np.arange(1., 9))
        for runner in (synthetic_run, real_run):
            row = runner(a, max_rank=6, block_size=2)
            self.assertEqual(row.rank, 1)
            self.assertEqual(row.inspections, 3)
            self.assertEqual(row.status, 'cap_reached')

    def test_q_remains_orthonormal(self):
        rng = np.random.default_rng(10)
        a = rng.standard_normal((9, 7))
        for module in (synthetic, real):
            q = np.empty((9, 0))
            for width in (3, 2):
                q = module.qb_step(rng, a, q, width)
            self.assertTrue(np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-12))

    def test_invalid_runner_inputs(self):
        for runner in (synthetic_run, real_run):
            for kwargs in (dict(block_size=0), dict(max_rank=0), dict(method='unknown'), dict(alpha=-1), dict(rel_tol=-1)):
                with self.assertRaises(ValueError):
                    runner(**kwargs)

    def test_adaptive_variant_records_requested_and_used_sizes(self):
        for runner in (synthetic_run, real_run):
            row = runner(adaptive_c=16)
            self.assertEqual(row.n_invalid_denominator, 0)
            self.assertEqual(row.s_used, row.s_requested)
            self.assertTrue(all(size >= 3 for size in row.s_used))


class CoverageDenominatorTests(unittest.TestCase):
    def test_undefined_rate_exports_as_na(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'summary.csv'
            coverage.write_dicts(output, [dict(rate=math.nan)])
            self.assertIn('NA', output.read_text())
            self.assertNotIn('nan', output.read_text())

    def test_no_certifications_has_undefined_conditional_rate(self):
        summary = coverage.summarize([synthetic_run()], 0.05)[0]
        self.assertEqual(summary['false_stop_rate'], 0)
        self.assertTrue(math.isnan(summary['conditional_false_stop_rate']))
        self.assertTrue(math.isnan(summary['conditional_false_stop_ci_high']))

    def test_conditional_and_unconditional_denominators(self):
        capped = synthetic_run()
        failure = dataclasses.replace(capped, certified=True, status='certified', certificate=0, false_stop=1)
        summary = coverage.summarize([capped, failure], 0.05)[0]
        self.assertEqual(summary['false_stop_rate'], 0.5)
        self.assertEqual(summary['conditional_false_stop_rate'], 1)
        self.assertEqual(summary['certified_runs'], 1)


if __name__ == '__main__':
    unittest.main()

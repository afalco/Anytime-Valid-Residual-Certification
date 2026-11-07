"""Separate R1 follow-up: explicit projector replication and implicit Schur cost."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, splu
from scipy.stats import chi2
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS / 'real_matrix_study'))
sys.path.insert(0, str(EXPERIMENTS / 'revision_R1_study' / 'audit'))
from run_real_matrices import qb_step, residual_certificate
from run_schedule_comparison import risk_level
from certification import inspection_risk
from cost_accounting import CostLedger, CoefficientCache, energy_residual


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SchurOperator(LinearOperator):
    """S = shift*I - C H^{-1} D, with sparse nonsymmetric coupling blocks."""

    def __init__(self, n, seed, shift):
        rng = np.random.default_rng(seed)
        weights = rng.uniform(0.8, 1.2, n + 1)
        self.h = (n * n * sparse.diags(
            [-weights[1:-1], weights[:-1] + weights[1:], -weights[1:-1]],
            [-1, 0, 1], shape=(n, n), format='csc')
            + sparse.diags(rng.uniform(0.5, 1.5, n), format='csc'))
        self.c = sparse.diags([rng.uniform(0.8, 1.2, n), rng.uniform(0.1, 0.2, n - 1)],
                              [0, -1], format='csr')
        self.d = sparse.diags([rng.uniform(0.8, 1.2, n), rng.uniform(-0.2, -0.1, n - 1)],
                              [0, 1], format='csr')
        self.shift = shift
        self.lu = splu(self.h)
        self.forward_rhs = self.transpose_rhs = 0
        super().__init__(dtype=np.dtype('float64'), shape=(n, n))

    def _matmat(self, x):
        self.forward_rhs += x.shape[1]
        return self.shift * x - self.c @ self.lu.solve(np.asarray(self.d @ x))

    def _rmatmat(self, x):
        self.transpose_rhs += x.shape[1]
        return self.shift * x - self.d.T @ self.lu.solve(np.asarray(self.c.T @ x), trans='T')

    def _matvec(self, x):
        return self._matmat(np.asarray(x).reshape(-1, 1)).ravel()

    def _rmatvec(self, x):
        return self._rmatmat(np.asarray(x).reshape(-1, 1)).ravel()


def columns(n, batch):
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        block = np.zeros((n, stop - start))
        block[np.arange(start, stop), np.arange(stop - start)] = 1
        yield start, stop, block


def column_norm(operator, ledger, phase, batch, q=None, bt=None):
    total = 0.0
    for start, stop, block in columns(operator.shape[1], batch):
        measured = ledger.apply(operator, block, phase)
        if q is not None:
            measured -= q @ bt[start:stop].T
        total += float(np.sum(measured * measured))
    return math.sqrt(total)


def implicit_trial(p, trial, method):
    start = perf_counter()
    operator = SchurOperator(p['n'], p['problem_seed'], p['shift'])
    setup_seconds = perf_counter() - start
    ledger = CostLedger()
    basis = np.random.default_rng([p['trial_seed'], trial, 0])
    probes = np.random.default_rng([p['trial_seed'], trial, 1])
    q = np.empty((p['n'], 0))
    cache = CoefficientCache(p['n'])
    norm_s = None
    if method == 'energy_scan':
        with ledger.measure('norm'):
            norm_s = column_norm(operator, ledger, 'norm', p['batch_size'])
    upper = math.inf
    invalid = 0
    horizon = math.ceil(p['max_rank'] / p['block_size'])
    prefix_checksums = []
    for j in range(1, horizon + 1):
        with ledger.measure('qb'):
            q = qb_step(basis, operator, q, min(p['block_size'], p['max_rank'] - q.shape[1]), costs=ledger)
        prefix_checksums.append(hashlib.sha256(q.tobytes()).hexdigest())
        if method == 'energy_scan':
            with ledger.measure('b'):
                cache.update(operator, q, ledger)
            with ledger.measure('certification'):
                residual, _ = energy_residual(norm_s, cache.bt)
                upper = residual * residual
        else:
            level = inspection_risk(p['alpha'], j, horizon, 'power')
            with ledger.measure('certification'):
                cert = residual_certificate(probes, operator, q, level, p['cert_size'], 1.,
                                            p['batch_size'], return_info=True, costs=ledger)
                upper = cert.upper
                invalid += int(not cert.denominator_valid)
        if upper <= p['epsilon'] ** 2 or q.shape[1] >= p['max_rank']:
            break
    if method != 'energy_scan':
        with ledger.measure('b'):
            cache.update(operator, q, ledger)
    solver_seconds = perf_counter() - start
    accepted = upper <= p['epsilon'] ** 2
    with ledger.measure('diagnostic'):
        residual = column_norm(operator, ledger, 'diagnostic', p['batch_size'], q, cache.bt)
    assert operator.forward_rhs == ledger.total('A') + ledger.total('A', diagnostic=True)
    assert operator.transpose_rhs == ledger.total('AT') + ledger.total('AT', diagnostic=True)
    return dict(trial=trial, method=method, rank=q.shape[1], inspections=j, certified=int(accepted),
                status='certified' if accepted else 'cap_reached', residual_ratio=residual / p['epsilon'],
                false_stop=int(accepted and residual > p['epsilon']), invalid=invalid,
                upper=upper if math.isfinite(upper) else 'Infinity',
                solver_A=ledger.total('A'), solver_AT=ledger.total('AT'), norm_A=ledger.counts['norm']['A'],
                qb_A=ledger.counts['qb']['A'], certification_A=ledger.counts['certification']['A'],
                diagnostic_A=ledger.total('A', diagnostic=True), diagnostic_AT=ledger.total('AT', diagnostic=True),
                setup_seconds=setup_seconds, solver_seconds=solver_seconds,
                costs_json=json.dumps(dict(counts=ledger.counts, seconds=ledger.seconds), sort_keys=True),
                prefix_checksums=prefix_checksums)


def explicit_check(p, target):
    records = []
    a = np.eye(p['n'])
    threshold = p['rel_tol'] ** 2 * p['n']
    # Precomputed quantiles change no random draws relative to the original check.
    quantiles = {(m, k): chi2.ppf(risk_level(m, k, p['alpha'], p['n']), p['s'] * (p['n'] - k)) /
                 (p['s'] * (p['n'] - k)) for m in ('naive', 'power', 'uniform') for k in range(1, p['n'])}
    for trial in range(p['trials']):
        for method in ('naive', 'power', 'uniform'):
            basis = np.random.default_rng([p['seed'], trial, 0])
            probes = np.random.default_rng([p['seed'], trial, 1])
            q = np.empty((p['n'], 0))
            for k in range(1, p['n'] + 1):
                q = qb_step(basis, a, q, 1)
                if q.shape[1] != k:
                    raise RuntimeError('unexpected rank breakdown on identity input')
                if k == p['n']:
                    upper = 0.
                else:
                    omega = probes.standard_normal((p['n'], p['s']))
                    sketch = omega - q @ (q.T @ omega)
                    upper = float(np.sum(sketch ** 2) / p['s']) / quantiles[method, k]
                if upper <= threshold:
                    break
            energy = float(np.sum((a - q @ q.T) ** 2))
            records.append(dict(trial=trial, method=method, rank=k, upper=upper,
                                numerical_residual_energy=energy, ideal_residual_energy=p['n'] - k,
                                certified=1, false_stop=int(energy > threshold), miscoverage=int(upper < p['n'] - k),
                                numerical_miscoverage=int(upper < energy),
                                orthogonality_error=float(np.linalg.norm(q.T @ q - np.eye(k), 'fro')),
                                full_rank_convention=int(k == p['n'])))
        if (trial + 1) % 250 == 0:
            print(f'Explicit check: {trial + 1}/{p["trials"]}', flush=True)
    write_csv(target / 'explicit_trials.csv', records)
    original = EXPERIMENTS / 'revision_R1_study/revision_R1/production_v1_full/exact_end_to_end/trials.csv'
    if original.exists():
        with original.open() as handle:
            previous = list(csv.DictReader(handle))
        for old, new in zip(previous, records):
            assert (int(old['trial']), old['method'], int(old['rank'])) == (new['trial'], new['method'], new['rank'])
            for field in ('upper', 'numerical_residual_energy', 'orthogonality_error'):
                assert math.isclose(float(old[field]), new[field], rel_tol=1e-12, abs_tol=1e-14)
        print('Verified original 500-trial prefix without changing historical files.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--group', choices=('explicit', 'implicit', 'all'), default='all')
    parser.add_argument('--pilot-trials', type=int)
    args = parser.parse_args()
    protocol = json.loads((HERE / 'protocol.json').read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = ('explicit_trials.csv', 'implicit_trials.jsonl') if args.group == 'all' else (
        ('explicit_trials.csv',) if args.group == 'explicit' else ('implicit_trials.jsonl',))
    if any((args.out_dir / name).exists() for name in names):
        parser.error('output already exists; choose a new directory to preserve previous runs')
    files = [Path(__file__), HERE / 'protocol.json', EXPERIMENTS / 'src/certification.py',
             EXPERIMENTS / 'src/cost_accounting.py', EXPERIMENTS / 'real_matrix_study/run_real_matrices.py',
             EXPERIMENTS / 'revision_R1_study/audit/run_schedule_comparison.py']
    manifest = dict(protocol=protocol, pilot_trials=args.pilot_trials, python=platform.python_version(),
                    numpy=np.__version__, scipy=scipy.__version__, threadpools=threadpool_info(),
                    source_sha256={str(f.relative_to(EXPERIMENTS)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (args.out_dir / f'manifest_{args.group}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with threadpool_limits(limits=protocol['blas_threads']):
        if args.group in ('all', 'explicit'):
            explicit_check(protocol['explicit_check'], args.out_dir)
        if args.group in ('all', 'implicit'):
            p = protocol['implicit_schur']
            count = p['trials'] if args.pilot_trials is None else args.pilot_trials
            if count < 1 or count > p['trials']:
                parser.error('pilot count must be between 1 and the frozen trial count')
            with (args.out_dir / 'implicit_trials.jsonl').open('w') as handle:
                for trial in range(count):
                    order = ['energy_scan', 'anytime_power']
                    np.random.default_rng([p['trial_seed'], trial, 2]).shuffle(order)
                    rows = []
                    for method in order:
                        row = implicit_trial(p, trial, method)
                        rows.append(row)
                        handle.write(json.dumps(row, allow_nan=False) + '\n')
                        handle.flush()
                        print('Schur:', trial, method, row['rank'], row['certified'], row['solver_A'], flush=True)
                    shared = min(len(r['prefix_checksums']) for r in rows)
                    assert rows[0]['prefix_checksums'][:shared] == rows[1]['prefix_checksums'][:shared]
    print('Follow-up complete.', flush=True)


if __name__ == '__main__':
    main()

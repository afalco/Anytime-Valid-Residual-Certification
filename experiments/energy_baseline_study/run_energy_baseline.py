"""Paired energy-identity versus Gaussian stopping with explicit access costs."""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS / 'real_matrix_study'))
sys.path.insert(0, str(EXPERIMENTS / 'synthetic_qb_study'))
from run_real_matrices import MatrixInfo, load_matrix, run_one
from run_synthetic_qb import synthetic_matrix


COMPARISONS = {
    'energy_incremental': ('energy', 'incremental'),
    'naive_final': ('naive', 'final'),
    'anytime_final': ('anytime', 'final'),
    'anytime_incremental': ('anytime', 'incremental'),
}


def write_rows(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), comparisons=COMPARISONS, python=platform.python_version(),
                  numpy=np.__version__, scipy=scipy.__version__,
                  thread_environment={key: os.environ.get(key) for key in
                                      ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
                  scope='stored matrices; direct offline diagnostics; pilot smoke study unless rerun at production scale')
    config['code_sha256'] = {str(path.relative_to(EXPERIMENTS)): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (Path(__file__), EXPERIMENTS / 'src' / 'certification.py',
                                          EXPERIMENTS / 'src' / 'cost_accounting.py',
                                          EXPERIMENTS / 'real_matrix_study' / 'run_real_matrices.py',
                                          EXPERIMENTS / 'synthetic_qb_study' / 'run_synthetic_qb.py')}
    config['data_sha256'] = {}
    for path in args.matrix_paths or []:
        with path.open('rb') as handle:
            config['data_sha256'][str(path)] = hashlib.file_digest(handle, 'sha256').hexdigest()
    (args.out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str) + '\n')
    rows = []
    inputs = args.matrix_paths or ['exponential', 'polynomial', 'flat_tail']
    for index, source in enumerate(inputs):
        loaded = load_matrix(source) if args.matrix_paths else None
        for trial in range(args.trials):
            streams = np.random.SeedSequence([args.seed, index, trial]).spawn(7)
            if loaded is None:
                a, _ = synthetic_matrix(np.random.default_rng(streams[0]), args.n, str(source))
                info = MatrixInfo(str(source), *a.shape, int(np.count_nonzero(a)), 0)
            else:
                a, info = loaded
            basis_seed = int(streams[1].generate_state(1)[0])
            order = np.random.default_rng(streams[2]).permutation(list(COMPARISONS))
            for sequence, comparison in enumerate(order):
                method, mode = COMPARISONS[comparison]
                seed_index = list(COMPARISONS).index(comparison)
                if comparison == 'anytime_incremental':
                    seed_index = list(COMPARISONS).index('anytime_final')
                cert_seed = int(streams[3 + seed_index].generate_state(1)[0])
                result = run_one(
                    rng=np.random.default_rng(cert_seed), basis_rng=np.random.default_rng(basis_seed),
                    a=a, info=info, method=method, trial=trial, block_size=args.block_size,
                    cert_size=args.cert_size, alpha=args.alpha, rel_tol=args.rel_tol,
                    max_rank=args.max_rank, rho_mode='conservative', cert_batch_size=args.cert_size,
                    rho_pilot_size=2, rho_pilot_shrink=0.5, coefficient_mode=mode,
                    diagnostic_mode='direct')
                rows.append(dict(comparison=str(comparison), basis_seed=basis_seed, certification_seed=cert_seed,
                                 execution_order=sequence, source=str(source), **asdict(result)))
    write_rows(args.out_dir / 'trials.csv', rows)
    summary = []
    for matrix in sorted({row['matrix'] for row in rows}):
        for comparison in COMPARISONS:
            group = [row for row in rows if row['matrix'] == matrix and row['comparison'] == comparison]
            record = dict(matrix=matrix, comparison=comparison, trials=len(group))
            for field in ('rank', 'inspections', 'solver_A', 'solver_AT', 'solver_runtime_sec',
                          'b_runtime_sec', 'certification_runtime_sec', 'residual_eval_runtime_sec',
                          'norm_runtime_sec', 'residual_ratio', 'certified', 'false_stop'):
                record['mean_' + field] = float(np.mean([row[field] for row in group]))
            summary.append(record)
    write_rows(args.out_dir / 'summary.csv', summary)
    print(f'wrote {len(rows)} trials and {len(summary)} summaries to {args.out_dir}')
    for row in summary:
        print(f"{row['matrix']} {row['comparison']}: k={row['mean_rank']:.2f}, "
              f"N_A={row['mean_solver_A']:.1f}, N_AT={row['mean_solver_AT']:.1f}, "
              f"accepted={row['mean_certified']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix-paths', nargs='+', type=Path)
    parser.add_argument('--n', type=int, default=80)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--block-size', type=int, default=5)
    parser.add_argument('--max-rank', type=int, default=60)
    parser.add_argument('--cert-size', type=int, default=80)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--rel-tol', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent / 'revision_R1' / 'pilot')
    args = parser.parse_args()
    if args.n < 12 or args.trials < 1:
        parser.error('n>=12 and trials>=1 are required')
    run(args)


if __name__ == '__main__':
    main()

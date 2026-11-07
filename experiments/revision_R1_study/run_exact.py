"""Reproduce exact-law results and an explicit adaptive-QB projection check."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import chi2
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'audit'))
sys.path.insert(0, str(HERE.parent / 'real_matrix_study'))
from run_schedule_comparison import run as exact_law, risk_level
from run_real_matrices import qb_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, default=HERE / 'revision_R1' / 'production_v1')
    args = parser.parse_args()
    protocol = json.loads((HERE / 'protocol.json').read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        p = protocol['exact']
        report = exact_law(argparse.Namespace(**p, out_dir=args.out_dir / 'exact'))
        print('Exact law:', [(r['method'], r['miscoverage_rate']) for r in report['methods']], flush=True)
        p = protocol['exact_end_to_end']
        records = []
        a = np.eye(p['n'])
        threshold = p['rel_tol']**2 * p['n']
        for trial in range(p['trials']):
            for method in ('naive', 'power', 'uniform'):
                # Same basis seed and same certification seed across policies in a trial.
                basis = np.random.default_rng([p['seed'], trial, 0])
                probes = np.random.default_rng([p['seed'], trial, 1])
                q = np.empty((p['n'], 0))
                for k in range(1, p['n'] + 1):
                    q = qb_step(basis, a, q, 1)
                    if q.shape[1] != k:
                        raise RuntimeError('unexpected rank breakdown on identity input')
                    if k == p['n']:
                        upper = 0.0  # Exact structural convention for the full orthogonal projector.
                    else:
                        omega = probes.standard_normal((p['n'], p['s']))
                        residual_sketch = omega - q @ (q.T @ omega)
                        z = float(np.sum(residual_sketch**2) / p['s'])
                        delta = risk_level(method, k, p['alpha'], p['n'])
                        df = p['s'] * (p['n'] - k)
                        upper = z / (chi2.ppf(delta, df) / df)
                    if upper <= threshold:
                        break
                residual_energy = float(np.sum((a - q @ q.T)**2))
                records.append(dict(trial=trial, method=method, rank=k, upper=upper,
                                    numerical_residual_energy=residual_energy, ideal_residual_energy=p['n']-k, certified=1,
                                    false_stop=int(residual_energy > threshold),
                                    miscoverage=int(upper < p['n']-k),
                                    numerical_miscoverage=int(upper < residual_energy),
                                    orthogonality_error=float(np.linalg.norm(q.T @ q - np.eye(k), 'fro')),
                                    full_rank_convention=int(k == p['n'])))
            if trial % 100 == 0:
                print(f'Explicit QB: {trial}/{p["trials"]}', flush=True)
        target = args.out_dir / 'exact_end_to_end'
        target.mkdir(exist_ok=True)
        with (target / 'trials.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (target / 'config.json').write_text(json.dumps(dict(protocol=p, analytical_law='chi2(s*(n-k))/(s*(n-k))',
                                                          zero_convention='full-rank projector in exact arithmetic'), indent=2) + '\n')
        print('Explicit QB done:', len(records), flush=True)


if __name__ == '__main__':
    main()

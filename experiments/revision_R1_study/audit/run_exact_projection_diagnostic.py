"""Exact-law simulation of fresh Gaussian certificates for A=I.

This is a distributional diagnostic, not a timed execution of a QB solver.
For a rank-k orthogonal projector, Z/||R||_F^2 is chi2(s*(n-k))/(s*(n-k)).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import chi2


def wilson(count: int, n: int) -> list[float]:
    z = 1.959963984540054
    rate = count / n
    den = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / den
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / den
    return [max(0, center - half), min(1, center + half)]


def run(args: argparse.Namespace) -> dict:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config['out_dir'] = str(args.out_dir)
    config.update(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    (args.out_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    # A third cumulant separates moment matching from equality in distribution.
    weights = np.array([0.9, 0.1])
    kappa_exact = float(8 * np.sum(weights**3) / args.s**2)
    nu = args.s / float(np.sum(weights**2))
    kappa_matched = 8 / nu**2
    assert not math.isclose(kappa_exact, kappa_matched, rel_tol=1e-10)

    # Check the spectrum underlying the exact projector law.
    rng = np.random.default_rng(args.seed + 1)
    q, _ = np.linalg.qr(rng.standard_normal((args.n, args.n // 4)))
    residual = np.eye(args.n) - q @ q.T
    singular = np.linalg.svd(residual, compute_uv=False)
    d = args.n - q.shape[1]
    assert np.allclose(singular[:d], 1) and np.allclose(singular[d:], 0, atol=1e-12)

    report = {
        'interpretation': 'Exact-law projector diagnostic; not a QB runtime benchmark.',
        'unequal_weight_third_cumulants': {'exact': kappa_exact, 'moment_matched': kappa_matched},
        'methods': [],
    }
    streams = np.random.SeedSequence(args.seed).spawn(2)
    for method, stream in zip(['naive', 'anytime'], streams):
        rng = np.random.default_rng(stream)
        active = np.ones(args.trials, dtype=bool)
        rank = np.full(args.trials, args.n, dtype=int)
        energy_final = np.zeros(args.trials)
        upper_final = np.zeros(args.trials)
        tolerance_sq = args.rel_tol**2 * args.n
        for k in range(1, args.n):
            indices = np.flatnonzero(active)
            if not indices.size:
                break
            energy = args.n - k
            df = args.s * energy
            level = args.alpha if method == 'naive' else 6 * args.alpha / (math.pi**2 * k**2)
            quantile = chi2.ppf(level, df) / df
            assert quantile > 0
            # A new draw at every inspection, independent of the approximation history.
            ratio = rng.chisquare(df, size=indices.size) / df
            upper = energy * ratio / quantile
            accepted = upper <= tolerance_sq
            stopped = indices[accepted]
            rank[stopped] = k
            energy_final[stopped] = energy
            upper_final[stopped] = upper[accepted]
            active[stopped] = False
        # At the full exact projector rank, R=0 and U=0 by convention.
        false = energy_final > tolerance_sq
        missed = upper_final < energy_final
        with (args.out_dir / f'{method}_trials.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['trial', 'rank', 'residual_energy', 'upper', 'certified', 'false_stop', 'miscoverage'])
            for i in range(args.trials):
                writer.writerow([i, int(rank[i]), energy_final[i], upper_final[i], 1, int(false[i]), int(missed[i])])
        report['methods'].append({
            'method': method,
            'trials': args.trials,
            'mean_rank': float(rank.mean()),
            'certified_rate': 1.0,
            'false_stop_rate': float(false.mean()),
            'false_stop_wilson95': wilson(int(false.sum()), args.trials),
            'miscoverage_rate': float(missed.mean()),
            'miscoverage_wilson95': wilson(int(missed.sum()), args.trials),
            'full_rank_returns': int(active.sum()),
        })
    (args.out_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=80)
    parser.add_argument('--s', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--rel-tol', type=float, default=0.7)
    parser.add_argument('--trials', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    if args.n < 4 or args.s < 1 or args.trials < 1 or not 0 < args.alpha < 1 or not 0 < args.rel_tol < 1:
        parser.error('Require n>=4, s>=1, trials>=1, and alpha and relative tolerance in (0,1).')
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()

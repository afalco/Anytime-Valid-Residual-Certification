"""Paired exact-law schedule audit, with analytic stopping probabilities."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import chi2, kstest

from run_exact_projection_diagnostic import wilson


METHODS = ('naive', 'power', 'uniform', 'geometric')


def risk_level(method: str, j: int, alpha: float, horizon: int) -> float:
    if method == 'naive':
        return alpha
    if method == 'power':
        return 6 * alpha / (math.pi**2 * j**2)
    if method == 'uniform':
        return alpha / horizon
    if method == 'geometric':
        return alpha * 0.5**j
    raise ValueError(method)


def analytic_metrics(method: str, n: int, s: int, alpha: float, rel_tol: float) -> dict:
    survival = 1.0
    mean_rank = false_stop = miscoverage = 0.0
    tol_sq = rel_tol**2 * n
    for k in range(1, n):
        energy = n - k
        df = s * energy
        delta = risk_level(method, k, alpha, n)
        q = chi2.ppf(delta, df) / df
        threshold = tol_sq * q / energy
        p_stop = float(chi2.cdf(df * threshold, df))
        p_stop_and_miss = float(chi2.cdf(df * min(q, threshold), df))
        mass = survival * p_stop
        mean_rank += k * mass
        if energy > tol_sq:
            false_stop += mass
        miscoverage += survival * p_stop_and_miss
        survival *= 1 - p_stop
    return dict(mean_rank=mean_rank + n * survival, false_stop_rate=false_stop,
                miscoverage_rate=miscoverage, full_rank_probability=survival)


def check_explicit_probes(args: argparse.Namespace, stream: np.random.SeedSequence) -> dict:
    rng = np.random.default_rng(stream)
    k = 30
    d = args.n - k
    q, _ = np.linalg.qr(rng.standard_normal((args.n, k)))
    ratios = np.empty(args.trials)
    for start in range(0, args.trials, 256):
        end = min(start + 256, args.trials)
        omega = rng.standard_normal((args.n, (end - start) * args.s))
        residual = omega - q @ (q.T @ omega)
        if start == 0:
            assert np.allclose(residual, (np.eye(args.n) - q @ q.T) @ omega, atol=1e-12)
        ratios[start:end] = np.sum(residual**2, axis=0).reshape(end - start, args.s).mean(axis=1) / d
    pit = chi2.cdf(args.s * d * ratios, args.s * d)
    test = kstest(pit, 'uniform')
    return {'n': args.n, 'k': k, 's': args.s, 'trials': args.trials,
            'ks_statistic': float(test.statistic), 'ks_pvalue': float(test.pvalue),
            'lower_tails': [{'nominal': a, 'count': int(np.sum(pit < a)),
                             'rate': float(np.mean(pit < a)),
                             'wilson95': wilson(int(np.sum(pit < a)), args.trials)}
                            for a in (0.01, 0.05, 0.10)]}


def run(args: argparse.Namespace) -> dict:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    streams = np.random.SeedSequence(args.seed).spawn(2)
    rng = np.random.default_rng(streams[0])
    config = vars(args).copy()
    config['out_dir'] = str(args.out_dir)
    config.update(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  design='paired: shared fresh rank-wise sketch energy, independent across ranks',
                  geometric_gamma=0.5, horizon=args.n)
    (args.out_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    states = {m: dict(active=np.ones(args.trials, dtype=bool),
                     rank=np.full(args.trials, args.n),
                     energy=np.zeros(args.trials), upper=np.zeros(args.trials)) for m in METHODS}
    tol_sq = args.rel_tol**2 * args.n
    schedule_rows = []
    for k in range(1, args.n + 1):
        for method in METHODS:
            schedule_rows.append(dict(method=method, inspection=k,
                                      alpha_j=risk_level(method, k, args.alpha, args.n)))
        if k == args.n:
            continue  # Known zero residual at full projector rank; no sketch is needed.
        energy = args.n - k
        df = args.s * energy
        # Counterfactual draws for every trial avoid active-set-dependent random streams.
        ratio = rng.chisquare(df, size=args.trials) / df
        for method, state in states.items():
            delta = risk_level(method, k, args.alpha, args.n)
            q = chi2.ppf(delta, df) / df
            assert 0 < q < 1
            upper = energy * ratio / q
            stop = state['active'] & (upper <= tol_sq)
            state['rank'][stop] = k
            state['energy'][stop] = energy
            state['upper'][stop] = upper[stop]
            state['active'][stop] = False
    with (args.out_dir / 'schedules.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['method', 'inspection', 'alpha_j'])
        writer.writeheader()
        writer.writerows(schedule_rows)
    results = []
    for method, state in states.items():
        false = state['energy'] > tol_sq
        missed = state['upper'] < state['energy']
        with (args.out_dir / f'{method}_trials.csv').open('w', newline='') as handle:
            fields = ['trial', 'rank', 'residual_energy', 'upper', 'status', 'tolerance_met',
                      'false_stop', 'miscoverage', 'full_rank_return']
            writer = csv.writer(handle)
            writer.writerow(fields)
            for i in range(args.trials):
                writer.writerow([i, int(state['rank'][i]), state['energy'][i], state['upper'][i],
                                 'certified', int(not false[i]), int(false[i]), int(missed[i]),
                                 int(state['active'][i])])
        exact = analytic_metrics(method, args.n, args.s, args.alpha, args.rel_tol)
        observed = dict(method=method, trials=args.trials, mean_rank=float(state['rank'].mean()),
                        certified_rate=1.0, false_stop_count=int(false.sum()),
                        false_stop_rate=float(false.mean()),
                        false_stop_wilson95=wilson(int(false.sum()), args.trials),
                        miscoverage_count=int(missed.sum()), miscoverage_rate=float(missed.mean()),
                        miscoverage_wilson95=wilson(int(missed.sum()), args.trials),
                        full_rank_returns=int(state['active'].sum()), analytic=exact)
        results.append(observed)
    differences = []
    for method in ('uniform', 'geometric'):
        delta_rank = states[method]['rank'] - states['power']['rank']
        differences.append(dict(comparison=f'{method} minus power',
                                mean_rank_difference=float(delta_rank.mean()),
                                paired_standard_error=float(delta_rank.std(ddof=1) / math.sqrt(args.trials))))
    report = dict(methods=results, paired_rank_differences=differences,
                  explicit_probe_check=check_explicit_probes(args, streams[1]),
                  warning='Exact-law diagnostic with known full-rank zero residual; no runtime or matvec benchmark.')
    (args.out_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=80)
    parser.add_argument('--s', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--rel-tol', type=float, default=0.7)
    parser.add_argument('--trials', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=4242)
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent / 'schedule_results')
    args = parser.parse_args()
    if args.n <= 30 or args.s < 1 or args.trials < 2 or not 0 < args.alpha <= 0.1 or not 0 < args.rel_tol < 1:
        parser.error('Require n>30, s>=1, trials>=2, alpha in (0,0.1], rel-tol in (0,1).')
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()

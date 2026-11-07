"""Validate complete R1 runs and generate tables, figures and an audit report."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

cache = Path(__file__).resolve().parent / 'revision_R1' / 'plot_cache'
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(cache / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(cache))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'audit'))
from run_exact_projection_diagnostic import wilson as _wilson
from run_production import GROUPS, expected, write_csv


NAMES = {'exponential': 'Exponential', 'polynomial': 'Polynomial', 'flat_tail': 'Flat tail',
         'noisy_low_rank': 'Noisy low rank', 'camera_like_192x192': 'Camera-like', 'digits_1797x64': 'Digits',
         'elliptic_laplacian_289x289': 'Laplacian', 'skimage_astronaut_gray_384x384': 'Astronaut',
         'skimage_camera_512x512': 'Camera', 'skimage_coins_303x384': 'Coins',
         'term_document_260x190': 'Term-doc', 'SNAP_ca-HepPh': 'ca-HepPh',
         'SNAP_email-Enron': 'email-Enron', 'SNAP_web-NotreDame': 'web-NotreDame',
         'LPnetlib_lp_fit2p_transpose': r'lp\_fit2p$^\top$'}


def name(value):
    return NAMES.get(value, value.replace('_', r'\_'))


def average(rows, field):
    return float(np.mean([float(row[field]) for row in rows]))


def wilson(count, total):
    lo, hi = _wilson(count, total)
    return (0.0 if count == 0 else lo, 1.0 if count == total else hi)


def probability(value):
    if value in (0, 1):
        return str(int(value))
    return f'{value:.6f}'.rstrip('0').rstrip('.')


def scientific_probability(value):
    mantissa, exponent = f'{value:.3e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'


def interval(count, total):
    if not total:
        return 'NA'
    lo, hi = wilson(count, total)
    return f'{probability(count/total)} [{probability(lo)},{probability(hi)}]'


def table(directory, filename, label, caption, headers, rows, append=False):
    text_columns = 2 if headers[1] in ('Method', 'Schedule', 'Rule', 'Size') else 1
    alignment = 'l' * text_columns + 'r' * (len(headers) - text_columns)
    lines = [r'\begin{table}[t]', r'  \centering', '  \\caption{' + caption + '}',
             '  \\label{' + label + '}', r'  \small', r'  \setlength{\tabcolsep}{4pt}',
             r'  \resizebox{\textwidth}{!}{%', '  \\begin{tabular}{' + alignment + '}',
             r'    \toprule', '    ' + ' & '.join(headers) + r' \\', r'    \midrule']
    lines.extend('    ' + ' & '.join(str(value) for value in row) + r' \\' for row in rows)
    lines.extend([r'    \bottomrule', r'  \end{tabular}}', r'\end{table}'])
    target = directory / filename
    temporary = target.with_suffix('.tmp.tex')
    prefix = target.read_text() + '\n' if append else ''
    temporary.write_text(prefix + '\n'.join(lines) + '\n')
    temporary.replace(target)


def save_figure(fig, target):
    temporary = target.with_suffix('.tmp.pdf')
    fig.savefig(temporary)
    temporary.replace(target)


def summarize(rows):
    n = len(rows)
    certified = sum(bool(r['certified']) for r in rows)
    failures = sum(int(r['false_stop']) for r in rows)
    misses = sum(int(r['miscoverage']) for r in rows)
    result = dict(group=rows[0]['group'], matrix=rows[0]['matrix'], variant=rows[0]['variant'], trials=n,
                  alpha=rows[0]['alpha'], cert_size=rows[0]['cert_size'], schedule=rows[0]['schedule'],
                  rows=rows[0]['rows'], cols=rows[0]['cols'], nnz=rows[0]['nnz'],
                  mean_rank=average(rows, 'rank'), rank_se=float(np.std([r['rank'] for r in rows], ddof=1) / math.sqrt(n)),
                  mean_inspections=average(rows, 'inspections'), certified_count=certified, certified_rate=certified/n,
                  false_stop_count=failures, false_stop_rate=failures/n, miscoverage_count=misses, miscoverage_rate=misses/n,
                  conditional_false_stop_rate=failures/certified if certified else 'NA',
                  mean_residual_ratio=average(rows, 'residual_ratio'),
                  invalid_inspection_fraction=sum(r['n_invalid_denominator'] for r in rows)/sum(r['inspections'] for r in rows),
                  infinite_final_fraction=sum(r['final_certificate_infinite'] for r in rows)/n,
                  mean_certification_probes=float(np.mean([sum(r['s_used']) for r in rows])))
    for field in ('solver_A', 'solver_AT', 'solver_runtime_sec', 'qb_runtime_sec', 'b_runtime_sec',
                  'certification_runtime_sec', 'residual_eval_runtime_sec', 'norm_runtime_sec'):
        if field in rows[0]:
            result['mean_' + field] = average(rows, field)
    for label, count, denominator in (('miscoverage', misses, n), ('false_stop', failures, n),
                                    ('conditional_false_stop', failures, certified)):
        limits = wilson(count, denominator) if denominator else ('NA', 'NA')
        result[label + '_ci_low'], result[label + '_ci_high'] = limits
    return result


def validate(root, protocol):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['protocol'] == protocol, 'protocol changed since execution'
    for filename, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((HERE.parent / filename).read_bytes()).hexdigest() == digest, filename
    for filename, digest in manifest['data_sha256'].items():
        with (Path(manifest['data_dir']) / filename).open('rb') as handle:
            assert hashlib.file_digest(handle, 'sha256').hexdigest() == digest, filename
    all_rows, report = [], {}
    for group in GROUPS:
        status = json.loads((root / group / 'status.json').read_text())
        rows = [json.loads(line) for line in (root / group / 'trials.jsonl').read_text().splitlines()]
        assert status['complete'] and len(rows) == expected(protocol, group), (group, len(rows))
        assert len({r['case_id'] for r in rows}) == len(rows)
        for r in rows:
            assert r['rank'] <= min(r['max_rank'], r['rows'], r['cols'])
            assert bool(r['false_stop']) == (bool(r['certified']) and not bool(r['tolerance_met']))
            assert r['status'] == ('certified' if r['certified'] else 'cap_reached')
            assert r['inspections'] == len(r['s_used']) == len(r['s_requested'])
            if 'costs_json' in r:
                counts = json.loads(r['costs_json'])['counts']
                assert counts['b']['AT'] == r['rank']
                assert counts['certification']['A'] == sum(r['s_used'])
                for direction in ('A', 'AT'):
                    assert r['solver_' + direction] == sum(c[direction] for p, c in counts.items() if p != 'diagnostic')
                assert r['runtime_sec'] >= r['solver_runtime_sec']
        report[group] = dict(trials=len(rows), complete=True)
        all_rows.extend(rows)
    return all_rows, report


def build(args):
    protocol = json.loads((HERE / 'protocol.json').read_text())
    rows, report = validate(args.results, protocol)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['group'], r['matrix'], r['variant'])].append(r)
    summaries = {key: summarize(value) for key, value in grouped.items()}
    write_csv(args.results / 'all_summaries.csv', list(summaries.values()))
    def get(group, matrix, variant):
        return summaries[(group, matrix, variant)]
    tables, figures = args.manuscript_dir / 'tables', args.manuscript_dir / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    spectra = protocol['spectra']

    coverage = []
    positivity = []
    for spectrum in spectra:
        for method in ('naive', 'anytime'):
            r = get('coverage', spectrum, method)
            coverage.append([name(spectrum), method, f"{r['mean_rank']:.1f}", f"{r['certified_count']}/{r['trials']}",
                             interval(r['miscoverage_count'], r['trials']), interval(r['false_stop_count'], r['trials'])])
        r = get('coverage', spectrum, 'anytime')
        positivity.append([name(spectrum), f"{100*r['invalid_inspection_fraction']:.1f}", f"{100*r['infinite_final_fraction']:.1f}",
                           r['certified_count'], interval(r['false_stop_count'], r['certified_count'])])
    table(tables, 'adaptive_qb_coverage_table.tex', 'tab:adaptive-qb-coverage',
          r'Revised LM diagnostic: $N=500$ per spectrum/method, $n=k_{\max}=80$, $b=1$, $s=3$, $\alpha=0.25$, '
          r'$\varepsilon=0.1\|A\|_F$, oracle effective rank and power spending. Miscoverage and false-stop rates '
          r'use all $N$ runs with Wilson 95\% intervals. Capped runs are included; no nominal undercoverage is inferred from nonzero rates alone.',
          ['Spectrum', 'Method', 'Mean rank', '$C/N$', 'Miscoverage [CI]', 'False stop [CI]'], coverage)
    table(tables, 'positivity_table.tex', 'tab:positivity-diagnostic',
          r'Positivity diagnostic for the anytime rows of Table~\ref{tab:adaptive-qb-coverage}. Invalid inspections and '
          r'infinite final certificates are percentages with different denominators (all inspections and all runs). '
          r'The final interval conditions descriptively on the $C$ certified outputs; it is NA when $C=0$. Infinite bounds give vacuous coverage.',
          ['Spectrum', r'Invalid inspections (\%)', r'Final $+\infty$ (\%)', '$C$', 'Conditional false stop [CI]'], positivity)

    exact = json.loads((args.results / 'exact' / 'summary.json').read_text())
    exact_rows = []
    for r in exact['methods']:
        n = r['trials']
        exact_rows.append([r['method'], f"{r['mean_rank']:.2f}", interval(round(n*r['miscoverage_rate']), n),
                           scientific_probability(r['analytic']['miscoverage_rate']), interval(round(n*r['false_stop_rate']), n),
                           scientific_probability(r['analytic']['false_stop_rate'])])
    full_rank_counts = ', '.join(f"{r['method']}: {r['full_rank_returns']}" for r in exact['methods'])
    table(tables, 'exact_projection_table.tex', 'tab:exact-projection',
          r'Exact-law projector diagnostic ($A=I_{80}$, $s=3$, $\alpha=0.05$, $\varepsilon=0.7\sqrt{80}$, $N=20000$, '
          r'seed 4242). MC columns have Wilson 95\% intervals; analytical columns use the chi-square CDF and survival recursion. '
          r'All runs certify; ideal full-rank zero returns are ' + full_rank_counts +
          r'. Analytical probabilities use scientific notation. This is not a matrix-free cost benchmark.',
          ['Rule', 'Rank', 'MC miscoverage [CI]', 'Analytical', 'MC false stop [CI]', 'Analytical'], exact_rows)
    with (args.results / 'exact_end_to_end' / 'trials.csv').open() as handle:
        explicit = list(csv.DictReader(handle))
    explicit_rows = []
    for method in ('naive', 'power', 'uniform'):
        group = [r for r in explicit if r['method'] == method]
        assert len(group) == protocol['exact_end_to_end']['trials']
        explicit_rows.append([method, f"{average(group, 'rank'):.2f}",
                              interval(sum(int(r['miscoverage']) for r in group), len(group)),
                              interval(sum(int(r['false_stop']) for r in group), len(group)),
                              sum(int(r['full_rank_convention']) for r in group)])
    table(tables, 'exact_qb_check_table.tex', 'tab:exact-qb-check',
          r'Explicit identity-matrix QB/probe check: 500 trials per rule with the settings of Table~\ref{tab:exact-projection}, '
          r'seed 20261108. Miscoverage targets the ideal projector energy $80-k$; the separately archived numerical residual '
          r'and orthogonality errors expose roundoff. Every output is accepted. The last column counts the structural full-rank convention.',
          ['Rule', 'Rank', 'Ideal miscoverage [CI]', 'False stop [CI]', 'Full rank'], explicit_rows)
    report['exact_end_to_end'] = dict(trials=len(explicit), max_orthogonality_error=max(float(r['orthogonality_error']) for r in explicit),
                                    full_rank_returns=sum(int(r['full_rank_convention']) for r in explicit))

    rank_rows, rho_rows = [], []
    for spectrum in spectra:
        n, p = get('rank', spectrum, 'naive'), get('rank', spectrum, 'anytime')
        rank_rows.append([name(spectrum), f"{n['mean_rank']:.2f}", f"{p['mean_rank']:.2f}",
                          f"{p['mean_rank']-n['mean_rank']:.2f}", f"{p['certified_count']}/40"])
        o, c = get('rho', spectrum, 'oracle'), get('rho', spectrum, 'conservative')
        rho_rows.append([name(spectrum), f"{o['mean_rank']:.2f}", f"{c['mean_rank']:.2f}",
                         f"{o['certified_count']}/30", f"{c['certified_count']}/30"])
    table(tables, 'synthetic_qb_overhead_table.tex', 'tab:synthetic-qb-overhead',
          r'Oracle rank diagnostic: $n=120$, $N=40$, $b=5$, $s=80$, $k_{\max}=100$, $\alpha=0.05$, '
          r'$\varepsilon=0.1\|A\|_F$, power spending. Each pair shares matrix, basis and certification seeds. '
          r'Ranks include cap exits; the last column reports certified anytime outputs.',
          ['Spectrum', 'Naive rank', 'Anytime rank', 'Difference', '$C/N$'], rank_rows)
    table(tables, 'effective_rank_sensitivity_table.tex', 'tab:effective-rank-sensitivity',
          r'Effective-rank information: $n=120$, $N=30$, $b=5$, $s=80$, $k_{\max}=110$, $\alpha=0.05$, '
          r'$\varepsilon=0.1\|A\|_F$, anytime power. Oracle rho is a spectral diagnostic, not an implementable lower-bound estimator.',
          ['Spectrum', 'Oracle rank', r'$\rho=1$ rank', 'Oracle $C/N$', r'$\rho=1$ $C/N$'], rho_rows)

    sensitivity = []
    adaptive = []
    for spectrum in protocol['sensitivity']['spectra']:
        for s in protocol['sensitivity']['sizes']:
            for schedule in ('power', 'uniform'):
                r = get('sensitivity', spectrum, f'{schedule}_s{s}_a0.05')
                sensitivity.append([name(spectrum), schedule, s, f"{r['mean_rank']:.1f}", f"{r['certified_count']}/100",
                                    f"{r['mean_solver_A']:.0f}", f"{r['mean_solver_AT']:.0f}", f"{100*r['invalid_inspection_fraction']:.1f}"])
        for schedule in ('power', 'uniform'):
            fixed = get('sensitivity', spectrum, f'{schedule}_s32_a0.05')
            a = get('adaptive', spectrum, schedule)
            adaptive.append([name(spectrum), schedule, f"{fixed['mean_rank']:.1f}/{a['mean_rank']:.1f}",
                             f"{fixed['mean_solver_A']:.0f}/{a['mean_solver_A']:.0f}",
                             f"${fixed['certified_count']}\\to{a['certified_count']}$", f"{100*a['invalid_inspection_fraction']:.1f}"])
    table(tables, 'sensitivity_table.tex', 'tab:sketch-risk-sensitivity',
          r'Central risk level $\alpha=0.05$ from the full prespecified sensitivity grid. $N=100$, $n=120$, '
          r'$b=5$, $k_{\max}=100$, $\rho=1$, $\varepsilon=0.1\|A\|_F$. Uniform spending uses $J=20$ inspections. '
          r'Costs include capped runs and final $B$; invalid is the percentage of attempted inspections. Other risk levels appear in Figure~\ref{fig:sketch-risk-sensitivity}.',
          ['Spectrum', 'Schedule', '$s$', 'Rank', '$C/N$', '$N_A$', r'$N_{A^\top}$', r'Invalid (\%)'], sensitivity)
    table(tables, 'adaptive_size_table.tex', 'tab:adaptive-size',
          r'Fixed $s=32$ versus adaptive $s_j=\max\{32,\lceil16\log(1/\alpha_j)/\rho_j\rceil\}$, '
          r'$\rho=1$, $\alpha=0.05$ and the remaining sensitivity settings. Slash-separated values are fixed/adaptive. '
          r'Fixed and adaptive groups use independent sets of 100 trials; arrows show fixed-to-adaptive certified counts, not fractions. '
          r'Positivity does not by itself imply certified termination.',
          ['Spectrum', 'Schedule', 'Rank F/A', '$N_A$ F/A', r'$C$: fixed $\to$ adaptive', r'Adaptive invalid (\%)'], adaptive)

    def cost_rows(group):
        output = []
        for matrix in sorted({k[1] for k in summaries if k[0] == group}):
            for variant in ('energy', 'naive', 'power', 'uniform', 'power_incremental'):
                if (group, matrix, variant) not in summaries:
                    continue
                r = get(group, matrix, variant)
                output.append([name(matrix), variant.replace('_', '+'), f"{r['mean_rank']:.1f}",
                               f"{r['certified_count']}/{r['trials']}", f"{r['mean_solver_A']:.0f}",
                               f"{r['mean_solver_AT']:.0f}", f"{r['mean_solver_runtime_sec']:.3g}"])
        return output
    cost_headers = ['Input', 'Rule', 'Rank', '$C/N$', '$N_A$', r'$N_{A^\top}$', 'Solver s']
    table(tables, 'synthetic_cost_table.tex', 'tab:synthetic-costs',
          r'Stored synthetic cost baseline: $N=40$, $n=120$, $b=5$, $s=80$, cap 100, $\alpha=0.05$, '
          r'$\varepsilon=0.1\|A\|_F$, $\rho=1$. Energy maintains $B$; stochastic rules form it at output except power+incremental. '
          r'Solver seconds include norm and coefficients, exclude offline checks, and are descriptive single-thread means, not universal speed claims.',
          cost_headers, cost_rows('cost'))
    stored_costs = cost_rows('real')
    for start in range(0, len(stored_costs), 12):
        label = 'tab:stored-costs' + (f'-{start//12+1}' if start else '')
        caption = (r'Stored benchmark/application cost baseline: $N=20$, $b=5$, $s=80$, cap $\min\{150,m,n\}$, '
                   r'$\alpha=0.05$, relative tolerance 0.1, $\rho=1$, paired basis prefixes. Energy maintains $B$; other rules form it at output. '
                   r'Uniform uses the declared inspection horizon. Solver time excludes direct stored-block diagnostics.') if not start else (
                   r'Continuation of Table~\ref{tab:stored-costs}, with the same protocol, units and timing convention.')
        table(tables, 'real_cost_table.tex', label, caption, cost_headers, stored_costs[start:start+12], append=bool(start))
    real_rows = []
    for matrix in sorted({k[1] for k in summaries if k[0] == 'real'}):
        values = [get('real', matrix, method) for method in ('energy', 'naive', 'power', 'uniform')]
        p, u = values[2:]
        real_rows.append([name(matrix), f"${p['rows']}\\times{p['cols']}$"] + [f"{v['mean_rank']:.1f}" for v in values] +
                         [f"{p['certified_count']}/20", f"{u['certified_count']}/20", f"{p['mean_residual_ratio']:.3f}"])
    table(tables, 'real_matrix_table.tex', 'tab:real-matrices',
          r'Paired stored-matrix results with the settings of Table~\ref{tab:stored-costs}. E/N/P/U denote energy, naive, power and uniform. '
          r'The final column is the power residual-to-tolerance ratio $\|A-QB\|_F/\varepsilon$; below one meets the tolerance but need not certify.',
          ['Input', 'Size', 'E rank', 'N rank', 'P rank', 'U rank', 'P $C/N$', 'U $C/N$', 'P residual/tol.'], real_rows)
    graph_rows, lp_rows = [], []
    for group in ('graphs', 'sparse_lowrank'):
        for matrix in sorted({k[1] for k in summaries if k[0] == group}):
            e, n, p = [get(group, matrix, v) for v in ('energy', 'naive', 'power')]
            target = graph_rows if group == 'graphs' else lp_rows
            target.append([name(matrix), f"${p['rows']}\\times{p['cols']}$", p['nnz'],
                           f"{e['mean_rank']:.1f}", f"{n['mean_rank']:.1f}", f"{p['mean_rank']:.1f}",
                           f"{p['certified_count']}/{p['trials']}", f"{p['mean_residual_ratio']:.3f}"])
    headers = ['Input', 'Size', 'nnz', 'E rank', 'N rank', 'P rank', 'P $C/N$', 'P residual/tol.']
    table(tables, 'suitesparse_matrix_table.tex', 'tab:suitesparse-matrices',
          r'SuiteSparse graph stress tests: $N=10$, $b=10$, $s=200$ in batches of 20, rank cap 120, '
          r'$\alpha=0.05$, $\rho=1$, $\varepsilon=0.95\|A\|_F$. E/N/P denote energy, naive and anytime power. '
          r'nnz is the number of nonzero entries. The ratio is $\|A-QB\|_F/\varepsilon$ (power), evaluated by cached energy, not a direct residual scan.', headers, graph_rows)
    table(tables, 'suitesparse_lowrank_table.tex', 'tab:suitesparse-lowrank',
          r'Compressible SuiteSparse LP transpose: $N=20$, $b=10$, $s=200$, cap 120, $\alpha=0.05$, $\rho=1$, '
          r'$\varepsilon=0.1\|A\|_F$. E/N/P denote energy, naive and power. Residual-to-tolerance ratios below one meet the tolerance; '
          r'a direct stored-block offline check is used. nnz counts nonzero entries.', headers, lp_rows)
    table(tables, 'suitesparse_cost_table.tex', 'tab:suitesparse-costs',
          r'Directional SuiteSparse solver costs for the settings of Tables~\ref{tab:suitesparse-matrices} and~\ref{tab:suitesparse-lowrank}. '
          r'Times include the stored norm pass and $B$ output, but exclude offline diagnostics. Zero extra products for cached energy do not imply zero arithmetic cost.',
          cost_headers, cost_rows('graphs') + cost_rows('sparse_lowrank'))
    phases = []
    for group in ('cost', 'real', 'graphs', 'sparse_lowrank'):
        for key in sorted(k for k in summaries if k[0] == group and k[2] == 'power'):
            r = summaries[key]
            phases.append([name(r['matrix'])] + [f"{r['mean_'+field]:.3g}" for field in
                          ('qb_runtime_sec', 'certification_runtime_sec', 'b_runtime_sec', 'solver_runtime_sec', 'residual_eval_runtime_sec')])
    table(tables, 'phase_cost_table.tex', 'tab:phase-costs',
          r'Anytime power timing decomposition in seconds (means over all runs, including caps). Solver time also includes input norm '
          r'and dispatch overhead; offline diagnostics are excluded. Synthetic cost runs use $\rho=1$, not oracle work.',
          ['Input', 'QB s', 'Cert. s', '$B$ s', 'Solver s', 'Offline s'], phases)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, field, title in zip(axes, ('miscoverage', 'false_stop'), ('Certificate miscoverage', 'Incorrect accepted output')):
        values = [r[field + '_rate'] for r in exact['methods']]
        bounds = [r[field + '_wilson95'] for r in exact['methods']]
        ax.bar(range(4), values, color=['#b64036', '#287d55', '#397cb9', '#8c667f'])
        ax.errorbar(range(4), values, yerr=np.array([[v-b[0] for v,b in zip(values,bounds)], [b[1]-v for v,b in zip(values,bounds)]]), fmt='none', color='black', capsize=3)
        ax.axhline(0.05, color='black', linestyle='--', linewidth=1, label='Nominal 0.05')
        ax.set_xticks(range(4), ['Naive', 'Power', 'Uniform', 'Geom.'], rotation=20)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Probability')
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, figures / 'exact_projection_coverage.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    x = np.arange(4)
    for offset, method, color in ((-0.18, 'naive', '#b64036'), (0.18, 'anytime', '#287d55')):
        selected = [get('rank', spectrum, method) for spectrum in spectra]
        ax.bar(x + offset, [r['mean_rank'] for r in selected], 0.34, yerr=[r['rank_se'] for r in selected], capsize=3, label=method, color=color)
    ax.set_xticks(x, [name(s) for s in spectra])
    ax.set_ylabel('Selected rank')
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figures / 'synthetic_qb_rank_overhead.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(7.8, 5.3))
    for row_index, spectrum in enumerate(protocol['sensitivity']['spectra']):
        for alpha, color in ((0.1, '#b64036'), (0.05, '#287d55'), (0.01, '#397cb9')):
            for schedule, style, marker in (('power', '-', 'o'), ('uniform', '--', 's')):
                data = [get('sensitivity', spectrum, f'{schedule}_s{s}_a{alpha}') for s in (32,80,200)]
                for column, field in enumerate(('mean_rank', 'mean_solver_A', 'certified_rate')):
                    axes[row_index,column].plot([32,80,200], [r[field] for r in data], style, marker=marker, color=color, linewidth=1.3,
                                                label=f'alpha={alpha}, {schedule}')
                    axes[row_index,column].set_xscale('log')
                    axes[row_index,column].set_xticks([32,80,200], ['32','80','200'])
                    axes[row_index,column].xaxis.set_minor_locator(NullLocator())
                    axes[row_index,column].xaxis.set_minor_formatter(NullFormatter())
                    axes[row_index,column].set_xlabel('Sketch size')
                    axes[row_index,column].grid(alpha=0.18)
        axes[row_index,0].set_ylabel(name(spectrum)+'\nMean rank')
        axes[row_index,1].set_ylabel('Mean applications of A')
        axes[row_index,2].set_ylabel('Certified fraction')
        axes[row_index,2].set_ylim(-0.04,1.04)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0,0.12,1,1))
    save_figure(fig, figures / 'sketch_risk_sensitivity.pdf')
    plt.close(fig)

    matrices = sorted({k[1] for k in summaries if k[0] == 'real'})
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    x = np.arange(len(matrices))
    for offset, method, color in ((-0.27,'energy','#777777'),(-0.09,'naive','#b64036'),(0.09,'power','#287d55'),(0.27,'uniform','#397cb9')):
        ax.bar(x+offset, [get('real',m,method)['mean_rank'] for m in matrices], width=0.17, label=method, color=color)
    ax.set_xticks(x, [name(m) for m in matrices], rotation=25, ha='right')
    ax.set_ylabel('Selected rank')
    ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5,1.15), fontsize=9)
    fig.tight_layout()
    save_figure(fig, figures / 'real_matrix_rank_comparison.pdf')
    plt.close(fig)

    report['total_method_trials'] = len(rows)
    report['protocol_sha256'] = hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()
    report['builder_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for path in tables.glob('*.tex'):
        assert all(byte >= 32 or byte == 10 for byte in path.read_bytes()), f'unexpected control character in {path}'
    report['artifacts'] = {str(path.relative_to(args.manuscript_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(args.manuscript_dir.rglob('*')) if path.is_file() and path.suffix in ('.tex','.pdf')}
    (args.results / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Validated {len(rows)} method trials; generated {len(report["artifacts"])} R1 resources')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=HERE / 'revision_R1' / 'production_v1_full')
    parser.add_argument('--manuscript-dir', type=Path, required=True)
    build(parser.parse_args())


if __name__ == '__main__':
    main()

"""Validate follow-up manifests and generate the implicit-operator table."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'revision_R1_study'))
from build_artifacts import table, interval


def validate_manifest(path, protocol):
    manifest = json.loads(path.read_text())
    assert manifest['protocol'] == protocol
    assert manifest['pilot_trials'] is None, 'pilot is not production evidence'
    for filename, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((HERE.parent / filename).read_bytes()).hexdigest() == digest, filename
    return manifest


def build(results, manuscript):
    protocol = json.loads((HERE / 'protocol.json').read_text())
    for group in ('explicit', 'implicit'):
        validate_manifest(results / f'manifest_{group}.json', protocol)
    with (results / 'explicit_trials.csv').open() as handle:
        explicit = list(csv.DictReader(handle))
    assert len(explicit) == 3 * protocol['explicit_check']['trials']
    explicit_summary = []
    for method in ('naive', 'power', 'uniform'):
        rows = [r for r in explicit if r['method'] == method]
        assert len(rows) == protocol['explicit_check']['trials']
        explicit_summary.append(dict(method=method, trials=len(rows), mean_rank=np.mean([int(r['rank']) for r in rows]),
                                     miscoverage=interval(sum(int(r['miscoverage']) for r in rows), len(rows)),
                                     false_stop=interval(sum(int(r['false_stop']) for r in rows), len(rows)),
                                     full_rank_returns=sum(int(r['full_rank_convention']) for r in rows)))
    rows = [json.loads(line) for line in (results / 'implicit_trials.jsonl').read_text().splitlines()]
    p = protocol['implicit_schur']
    assert len(rows) == 2 * p['trials']
    assert len({(r['trial'], r['method']) for r in rows}) == len(rows)
    for row in rows:
        assert row['rank'] <= p['max_rank']
        assert row['solver_AT'] == row['rank']
        assert row['solver_A'] == row['norm_A'] + row['qb_A'] + row['certification_A']
        assert row['norm_A'] == (p['n'] if row['method'] == 'energy_scan' else 0)
        assert row['diagnostic_A'] == p['n'] and row['diagnostic_AT'] == 0
        assert row['false_stop'] == int(row['certified'] and row['residual_ratio'] > 1)
    for trial in range(p['trials']):
        paired = [r for r in rows if r['trial'] == trial]
        assert len(paired) == 2
        shared = min(len(r['prefix_checksums']) for r in paired)
        assert paired[0]['prefix_checksums'][:shared] == paired[1]['prefix_checksums'][:shared]
    summaries, tex_rows = [], []
    for method in ('energy_scan', 'anytime_power'):
        selected = [r for r in rows if r['method'] == method]
        summary = dict(method=method, trials=len(selected), certified=sum(r['certified'] for r in selected),
                       false_stop=interval(sum(r['false_stop'] for r in selected), len(selected)))
        for field in ('rank', 'solver_A', 'solver_AT', 'norm_A', 'qb_A', 'certification_A',
                      'diagnostic_A', 'residual_ratio', 'setup_seconds', 'solver_seconds'):
            summary[field] = float(np.mean([r[field] for r in selected]))
        summary['phase_seconds'] = {phase: float(np.mean([json.loads(r['costs_json'])['seconds'][phase] for r in selected]))
                                    for phase in ('norm', 'qb', 'b', 'certification', 'diagnostic')}
        summaries.append(summary)
        tex_rows.append(['Energy scan' if method == 'energy_scan' else 'Anytime power', f"{summary['rank']:.2f}",
                         f"{summary['certified']}/{len(selected)}", f"{summary['norm_A']:.0f}",
                         f"{summary['solver_A']:.1f}", f"{summary['solver_AT']:.1f}",
                         f"{summary['solver_seconds']:.4f}", f"{summary['residual_ratio']:.3f}"])
    output = manuscript / 'tables'
    table(output, 'implicit_schur_table.tex', 'tab:implicit-schur',
          r'Implicit Schur complement: $n=1024$, $N=20$ paired trials per rule, $b=5$, cap 60, '
          r'$\varepsilon=0.01$ (absolute), $s=80$, $\rho=1$, $\alpha=0.05$. '
          r'Each forward/transpose RHS requires one $H$/$H^\top$ solve. $N_{\rm norm}$ is included in $N_S$. '
          r'Energy scan evaluates all 1024 columns once per run without storing $S$; anytime uses no input norm. '
          r'Solver seconds include setup and output coefficients, exclude the separate 1024-RHS direct residual check. '
          r'The final column is $\|S-QB\|_F/\varepsilon$. Energy stopping is a float64 baseline, not an outward-rounded certificate.',
          ['Rule', 'Rank', '$C/N$', r'$N_{\rm norm}$', '$N_S$', r'$N_{S^\top}$', 'Solver s', 'Residual/tol.'], tex_rows)
    report = dict(protocol=protocol, explicit_check=explicit_summary, implicit_schur=summaries,
                  builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  files_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in results.iterdir()
                                if f.suffix in ('.csv', '.jsonl')},
                  table_sha256=hashlib.sha256((output / 'implicit_schur_table.tex').read_bytes()).hexdigest())
    (results / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--manuscript-dir', type=Path, required=True)
    args = parser.parse_args()
    build(args.results, args.manuscript_dir)

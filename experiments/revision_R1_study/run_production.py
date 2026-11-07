"""Frozen, checkpointed R1 experiments. Scientific parameters live in protocol.json."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
for directory in ('src', 'real_matrix_study', 'synthetic_qb_study'):
    sys.path.insert(0, str(EXPERIMENTS / directory))
from run_real_matrices import MatrixInfo, load_matrix, run_one as run_cost
from run_synthetic_qb import synthetic_matrix, run_one as run_diagnostic

GROUPS = ('coverage', 'rank', 'rho', 'cost', 'sensitivity', 'adaptive', 'real', 'sparse_lowrank', 'graphs')


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 'NA' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean(row) for row in rows)


def policies(group, p):
    if group in ('coverage', 'rank'):
        return [dict(variant=m, method=m, rho_mode=p['rho_mode']) for m in ('naive', 'anytime')]
    if group == 'rho':
        return [dict(variant=r, method='anytime', rho_mode=r) for r in ('oracle', 'conservative')]
    if group == 'sensitivity':
        return [dict(variant=f'{schedule}_s{s}_a{alpha}', method='anytime', schedule=schedule,
                     cert_size=s, alpha=alpha) for alpha in p['alphas'] for s in p['sizes'] for schedule in p['schedules']]
    if group == 'adaptive':
        return [dict(variant=schedule, method='anytime', schedule=schedule, adaptive_c=p['adaptive_c']) for schedule in p['schedules']]
    result = [dict(variant='energy', method='energy', coefficient_mode='incremental'),
              dict(variant='naive', method='naive'), dict(variant='power', method='anytime')]
    if group == 'cost':
        result.append(dict(variant='power_incremental', method='anytime', coefficient_mode='incremental'))
    if group == 'real':
        result.append(dict(variant='uniform', method='anytime', schedule='uniform'))
    return result


def cases(protocol, group, data_dir):
    p = protocol[group]
    sources = p.get('files', p.get('spectra', protocol['spectra']))
    for source_index, source in enumerate(sources):
        loaded = load_matrix(data_dir / source) if 'files' in p else None
        for trial in range(p['trials']):
            # Deliberately independent of schedule, sketch size, alpha, and coefficient mode.
            matrix_seed = int(np.random.SeedSequence([protocol['seed'], GROUPS.index(group), source_index, trial, 0]).generate_state(1)[0])
            basis_seed = int(np.random.SeedSequence([protocol['seed'], GROUPS.index(group), source_index, trial, 1]).generate_state(1)[0])
            cert_seed = int(np.random.SeedSequence([protocol['seed'], GROUPS.index(group), source_index, trial, 2]).generate_state(1)[0])
            if loaded is None:
                a, _ = synthetic_matrix(np.random.default_rng(matrix_seed), p['n'], source)
                info = MatrixInfo(source, *a.shape, int(np.count_nonzero(a)), 0)
            else:
                a, info = loaded
            variants = policies(group, p)
            order_rng = np.random.default_rng([protocol['seed'], GROUPS.index(group), source_index, trial, 3])
            for ordinal, index in enumerate(order_rng.permutation(len(variants))):
                policy = variants[index]
                identity = f'{group}/{source_index}/{trial}/{policy["variant"]}'
                yield identity, a, info, trial, policy, dict(matrix_seed=matrix_seed, basis_seed=basis_seed,
                                                           certification_seed=cert_seed, execution_order=ordinal, source=source)


def expected(protocol, group):
    p = protocol[group]
    return len(p.get('files', p.get('spectra', protocol['spectra']))) * p['trials'] * len(policies(group, p))


def execute(protocol, group, a, info, trial, policy, seeds):
    p = protocol[group]
    core = dict(rng=np.random.default_rng(seeds['certification_seed']), basis_rng=np.random.default_rng(seeds['basis_seed']),
                a=a, method=policy['method'], trial=trial, block_size=p['block_size'],
                cert_size=policy.get('cert_size', p.get('cert_size', 80)), alpha=policy.get('alpha', p.get('alpha', 0.05)),
                rel_tol=p['rel_tol'], max_rank=p['max_rank'], rho_mode=policy.get('rho_mode', 'conservative'),
                schedule=policy.get('schedule', 'power'), adaptive_c=policy.get('adaptive_c'))
    if group in ('coverage', 'rank', 'rho'):
        result = run_diagnostic(spectrum=info.name, **core)
        row = asdict(result)
        row.update(matrix=info.name, residual_ratio=result.residual_fro / result.tolerance,
                   certificate_ratio=math.sqrt(result.certificate) / result.tolerance,
                   miscoverage=int(result.certificate < result.residual_fro**2))
    else:
        result = run_cost(info=info, cert_batch_size=p.get('cert_batch_size', 40),
                          rho_pilot_size=2, rho_pilot_shrink=0.5,
                          coefficient_mode=policy.get('coefficient_mode', 'final'),
                          diagnostic_mode='energy' if group == 'graphs' else 'direct', **core)
        row = asdict(result)
        row['miscoverage'] = int(result.certificate_ratio < result.residual_ratio)
    row.update(group=group, variant=policy['variant'], alpha=core['alpha'], cert_size=core['cert_size'],
               schedule=core['schedule'], rho_mode=core['rho_mode'], block_size=p['block_size'],
               max_rank=p['max_rank'], rel_tol=p['rel_tol'], adaptive_c=core['adaptive_c'],
               rows=a.shape[0], cols=a.shape[1], nnz=info.nnz, **seeds)
    return row


def provenance(args, protocol):
    files = [Path(__file__), HERE / 'protocol.json', EXPERIMENTS / 'src' / 'certification.py',
             EXPERIMENTS / 'src' / 'cost_accounting.py',
             EXPERIMENTS / 'real_matrix_study' / 'run_real_matrices.py',
             EXPERIMENTS / 'synthetic_qb_study' / 'run_synthetic_qb.py', HERE / 'run_exact.py',
             HERE / 'audit' / 'run_schedule_comparison.py', HERE / 'audit' / 'run_exact_projection_diagnostic.py']
    source = {str(path.relative_to(EXPERIMENTS)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    data = {}
    for group in ('real', 'sparse_lowrank', 'graphs'):
        for filename in protocol[group]['files']:
            with (args.data_dir / filename).open('rb') as handle:
                data[filename] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return dict(protocol=protocol, source_sha256=source, python=platform.python_version(), numpy=np.__version__,
                scipy=scipy.__version__, threadpools=threadpool_info(), data_sha256=data, data_dir=str(args.data_dir.resolve()))


def run(args):
    protocol = json.loads((HERE / 'protocol.json').read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / 'manifest.json'
    with threadpool_limits(limits=protocol['blas_threads']):
        manifest = provenance(args, protocol)
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous != manifest:
                raise ValueError('provenance differs; use a new output directory instead of overwriting results')
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        for group in args.groups:
            directory = args.out_dir / group
            directory.mkdir(exist_ok=True)
            checkpoint = directory / 'trials.jsonl'
            rows = [json.loads(line) for line in checkpoint.read_text().splitlines()] if checkpoint.exists() else []
            completed = {row['case_id'] for row in rows}
            if len(completed) != len(rows):
                raise ValueError('duplicate checkpoint case identifiers')
            performed = 0
            start = perf_counter()
            for identity, a, info, trial, policy, seeds in cases(protocol, group, args.data_dir):
                if identity in completed:
                    continue
                if args.max_cases is not None and performed >= args.max_cases:
                    break
                row = clean(dict(case_id=identity, **execute(protocol, group, a, info, trial, policy, seeds)))
                with checkpoint.open('a') as handle:
                    handle.write(json.dumps(row, allow_nan=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                rows.append(row)
                performed += 1
                if performed == 1 or performed % 50 == 0 or group == 'graphs':
                    print(f'{group}: {len(rows)}/{expected(protocol, group)}; last={identity}; {perf_counter()-start:.1f}s', flush=True)
            write_csv(directory / 'trials.csv', rows)
            status = dict(group=group, completed=len(rows), expected=expected(protocol, group),
                          complete=len(rows) == expected(protocol, group), wall_seconds_this_invocation=perf_counter()-start)
            (directory / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
            print(status, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, default=HERE / 'revision_R1' / 'production_v1')
    parser.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    parser.add_argument('--max-cases', type=int, default=None, help='bounded preflight; incomplete groups remain explicitly marked')
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error('max-cases must be positive')
    run(args)


if __name__ == '__main__':
    main()

"""Run adaptive QB stopping rules on local real matrices.

Supported input formats:

- .npy dense NumPy arrays
- .npz dense NumPy archives or scipy.sparse.save_npz files
- .mtx Matrix Market files
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import sys
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from certification import CertificateResult, extend_basis, gaussian_certificate, positive_int, sketch_size, tolerance_ratio, validate_run, inspection_risk
from cost_accounting import CostLedger, CoefficientCache, PHASES, direct_stored_residual, energy_residual

try:
    import scipy.io
    import scipy.sparse as sp
except Exception:  # pragma: no cover - used only when scipy is unavailable
    scipy = None
    sp = None


@dataclass(frozen=True)
class MatrixInfo:
    name: str
    rows: int
    cols: int
    nnz: int
    sparse: int


@dataclass(frozen=True)
class TrialResult:
    matrix: str
    rows: int
    cols: int
    nnz: int
    sparse: int
    method: str
    trial: int
    rank: int
    inspections: int
    residual_ratio: float
    certificate_ratio: float
    runtime_sec: float
    qb_runtime_sec: float
    certification_runtime_sec: float
    residual_eval_runtime_sec: float
    qb_matvecs: int
    certification_matvecs: int
    residual_eval_matvecs: int
    total_matvecs: int
    false_stop: int
    status: str
    certified: bool
    tolerance_met: bool
    n_invalid_denominator: int
    final_certificate_infinite: bool
    s_used: tuple[int, ...]
    s_requested: tuple[int, ...]
    solver_runtime_sec: float
    b_runtime_sec: float
    pilot_runtime_sec: float
    oracle_runtime_sec: float
    norm_runtime_sec: float
    solver_A: int
    solver_AT: int
    diagnostic_A: int
    diagnostic_AT: int
    norm_entries_read: int
    diagnostic_entries_materialized: int
    oracle_entries_materialized: int
    coefficient_mode: str
    diagnostic_mode: str
    certificate_kind: str
    energy_clipped: bool
    costs_json: str
    schedule: str


def alpha_spending(alpha: float, inspection: int) -> float:
    return 6.0 * alpha / (math.pi**2 * inspection**2)


def is_sparse_matrix(a: Any) -> bool:
    return sp is not None and sp.issparse(a)


def matmul(a: Any, x: np.ndarray) -> np.ndarray:
    y = a @ x
    return np.asarray(y)


def load_matrix(path: Path) -> tuple[Any, MatrixInfo]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        a = np.load(path)
    elif suffix == ".npz":
        if sp is not None:
            try:
                a = sp.load_npz(path)
            except Exception:
                loaded = np.load(path)
                if len(loaded.files) == 1:
                    a = loaded[loaded.files[0]]
                else:
                    raise ValueError(f"dense npz must contain one array: {path}")
        else:
            loaded = np.load(path)
            if len(loaded.files) == 1:
                a = loaded[loaded.files[0]]
            else:
                raise ValueError(f"dense npz must contain one array: {path}")
    elif suffix == ".mtx":
        if scipy is None:
            raise RuntimeError("scipy is required to load .mtx files")
        a = scipy.io.mmread(path)
    else:
        raise ValueError(f"unsupported matrix format: {path}")

    if is_sparse_matrix(a):
        a = a.astype(float).tocsr()
        rows, cols = a.shape
        info = MatrixInfo(path.stem, rows, cols, int(a.nnz), 1)
        return a, info

    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"matrix must be 2D: {path}")
    rows, cols = a.shape
    info = MatrixInfo(path.stem, rows, cols, int(np.count_nonzero(a)), 0)
    return a, info


def fro_norm(a: Any) -> float:
    if is_sparse_matrix(a):
        return float(np.sqrt(a.multiply(a).sum()))
    return float(np.linalg.norm(a, "fro"))


def residual_product(a: Any, q: np.ndarray, omega: np.ndarray, *, costs=None, phase='certification') -> np.ndarray:
    y = matmul(a, omega) if costs is None else costs.apply(a, omega, phase)
    if q.size:
        y = y - q @ (q.T @ y)
    return y


def residual_fro_exact(a: Any, q: np.ndarray, norm_a: float) -> float:
    if q.size == 0:
        return norm_a
    aq = matmul(a.T, q)
    captured = float(np.sum(aq * aq))
    residual_sq = max(norm_a * norm_a - captured, 0.0)
    return math.sqrt(residual_sq)


def effective_rank_oracle(a: Any, q: np.ndarray) -> float:
    rows, cols = a.shape
    if rows * cols > 4_000_000:
        raise RuntimeError(
            "oracle effective-rank mode would densify a large matrix; "
            "use --rho-mode conservative or --rho-mode pilot"
        )
    if is_sparse_matrix(a):
        dense = a.toarray()
    else:
        dense = np.asarray(a)
    if q.size:
        dense = dense - q @ (q.T @ dense)
    singular = np.linalg.svd(dense, compute_uv=False)
    energy = float(np.sum(singular**2))
    fourth = float(np.sum(singular**4))
    if energy == 0.0 or fourth == 0.0:
        return 1.0
    return (energy * energy) / fourth


def residual_certificate(
    rng: np.random.Generator,
    a: Any,
    q: np.ndarray,
    alpha_level: float,
    cert_size: int,
    rho: float,
    cert_batch_size: int,
    *,
    known_zero: bool = False,
    return_info: bool = False,
    costs=None,
) -> float | CertificateResult:
    result = gaussian_certificate(rng, lambda omega: residual_product(a, q, omega, costs=costs),
                                  a.shape[1], alpha_level, cert_size, rho,
                                  cert_batch_size, known_zero=known_zero)
    return result if return_info else result.upper


def effective_rank_pilot(
    rng: np.random.Generator,
    a: Any,
    q: np.ndarray,
    pilot_size: int,
    shrink: float,
    *,
    costs=None,
) -> float:
    omega = rng.standard_normal((a.shape[1], pilot_size))
    residual_sketch = residual_product(a, q, omega, costs=costs, phase='pilot')
    trace_estimate = float(np.sum(residual_sketch * residual_sketch) / pilot_size)
    if trace_estimate <= 0.0:
        return 1.0
    normal_sketch = matmul(a.T, residual_sketch) if costs is None else costs.apply(a, residual_sketch, 'pilot', transpose=True)
    fourth_estimate = float(np.sum(normal_sketch * normal_sketch) / pilot_size)
    if fourth_estimate <= 0.0:
        return 1.0
    return max(1.0, shrink * trace_estimate * trace_estimate / fourth_estimate)


def qb_step(rng: np.random.Generator, a: Any, q: np.ndarray, block_size: int, *, costs=None) -> np.ndarray:
    omega = rng.standard_normal((a.shape[1], block_size))
    y = matmul(a, omega) if costs is None else costs.apply(a, omega, 'qb')
    return extend_basis(q, y, float(np.linalg.norm(y)))


def run_one(
    rng: np.random.Generator,
    a: Any,
    info: MatrixInfo,
    method: str,
    trial: int,
    block_size: int,
    cert_size: int,
    alpha: float,
    rel_tol: float,
    max_rank: int,
    rho_mode: str,
    cert_batch_size: int,
    rho_pilot_size: int,
    rho_pilot_shrink: float,
    adaptive_c: float | None = None,
    coefficient_mode: str = 'final',
    diagnostic_mode: str = 'energy',
    basis_rng: np.random.Generator | None = None,
    schedule: str = 'power',
) -> TrialResult:
    validate_run('naive' if method == 'energy' else method, rho_mode, ('conservative', 'oracle', 'pilot'), block_size,
                 max_rank, rel_tol, alpha, cert_size)
    if coefficient_mode not in ('final', 'incremental') or diagnostic_mode not in ('energy', 'direct'):
        raise ValueError('unknown coefficient or diagnostic mode')
    if method == 'energy' and coefficient_mode != 'incremental':
        raise ValueError('energy stopping requires incremental coefficients')
    sketch_size(alpha, cert_size, 1.0, adaptive_c)
    inspection_risk(alpha, 1, 1, schedule)
    positive_int(cert_batch_size, 'cert_batch_size')
    positive_int(rho_pilot_size, 'rho_pilot_size')
    if not math.isfinite(rho_pilot_shrink) or not 0 < rho_pilot_shrink <= 1:
        raise ValueError('rho_pilot_shrink must be in (0,1]')
    values = a.data if is_sparse_matrix(a) else np.asarray(a)
    if len(a.shape) != 2 or min(a.shape) < 1 or not np.isfinite(values).all():
        raise ValueError('a must be a nonempty finite matrix')
    start = time.perf_counter()
    costs = CostLedger()
    basis_rng = rng if basis_rng is None else basis_rng
    q = np.empty((a.shape[0], 0))
    with costs.measure('norm'):
        norm_a = fro_norm(a)
    tolerance = rel_tol * norm_a
    if not math.isfinite(norm_a) or not math.isfinite(tolerance):
        raise ValueError('matrix norm and tolerance must be finite')
    inspection = 0
    certificate = math.inf
    cache = CoefficientCache(a.shape[1])
    energy_clipped = False
    oracle_entries = 0
    cap = min(max_rank, *a.shape)
    horizon = math.ceil(cap / block_size)
    known_zero = not np.any(values)
    invalid = 0
    used, requested = [], []

    while q.shape[1] < cap and inspection < horizon:
        width = min(block_size, cap - q.shape[1])
        with costs.measure('qb'):
            q = qb_step(basis_rng, a, q, width, costs=costs)
        inspection += 1
        if coefficient_mode == 'incremental':
            with costs.measure('b'):
                cache.update(a, q, costs)
        if method == 'energy':
            with costs.measure('certification'):
                energy, clipped = energy_residual(norm_a, cache.bt)
                energy_clipped |= clipped
                certificate = energy * energy
            used.append(0)
            requested.append(0)
        else:
            if known_zero or rho_mode == 'conservative':
                rho = 1.0
            elif rho_mode == 'oracle':
                with costs.measure('oracle'):
                    rho = max(1.0, effective_rank_oracle(a, q))
                    oracle_entries += a.shape[0] * a.shape[1]
            else:
                with costs.measure('pilot'):
                    rho = effective_rank_pilot(rng, a, q, rho_pilot_size, rho_pilot_shrink, costs=costs)
            alpha_level = alpha if method == 'naive' else inspection_risk(alpha, inspection, horizon, schedule)
            size = sketch_size(alpha_level, cert_size, rho, adaptive_c)
            with costs.measure('certification'):
                result = residual_certificate(rng, a, q, alpha_level, size, rho, cert_batch_size,
                                              known_zero=known_zero, return_info=True, costs=costs)
            certificate = result.upper
            invalid += int(not result.denominator_valid)
            used.append(result.probes_used)
            requested.append(size)
        if certificate <= tolerance * tolerance:
            break

    # The output is a QB factorization: charge its coefficients, not an extra diagnostic A^T Q.
    if coefficient_mode == 'final':
        with costs.measure('b'):
            cache.update(a, q, costs)
    solver_elapsed = time.perf_counter() - start
    with costs.measure('diagnostic'):
        if diagnostic_mode == 'direct':
            residual = direct_stored_residual(a, q, cache.bt)
        else:
            residual, clipped = energy_residual(norm_a, cache.bt)
            energy_clipped |= clipped
    elapsed = time.perf_counter() - start
    certified = certificate <= tolerance * tolerance
    tolerance_met = residual <= tolerance
    return TrialResult(
        matrix=info.name,
        rows=info.rows,
        cols=info.cols,
        nnz=info.nnz,
        sparse=info.sparse,
        method=method,
        trial=trial,
        rank=int(q.shape[1]),
        inspections=inspection,
        residual_ratio=tolerance_ratio(residual, tolerance),
        certificate_ratio=tolerance_ratio(math.sqrt(certificate), tolerance),
        runtime_sec=elapsed,
        qb_runtime_sec=costs.seconds['qb'],
        certification_runtime_sec=costs.seconds['certification'],
        residual_eval_runtime_sec=costs.seconds['diagnostic'],
        qb_matvecs=costs.counts['qb']['A'],
        certification_matvecs=costs.counts['certification']['A'],
        residual_eval_matvecs=costs.total('A', diagnostic=True) + costs.total('AT', diagnostic=True),
        total_matvecs=costs.total('A') + costs.total('AT') + costs.total('A', diagnostic=True) + costs.total('AT', diagnostic=True),
        false_stop=int(certified and not tolerance_met),
        status='certified' if certified else 'cap_reached',
        certified=certified,
        tolerance_met=tolerance_met,
        n_invalid_denominator=invalid,
        final_certificate_infinite=math.isinf(certificate),
        s_used=tuple(used),
        s_requested=tuple(requested),
        solver_runtime_sec=solver_elapsed,
        b_runtime_sec=costs.seconds['b'],
        pilot_runtime_sec=costs.seconds['pilot'],
        oracle_runtime_sec=costs.seconds['oracle'],
        norm_runtime_sec=costs.seconds['norm'],
        solver_A=costs.total('A'),
        solver_AT=costs.total('AT'),
        diagnostic_A=costs.total('A', diagnostic=True),
        diagnostic_AT=costs.total('AT', diagnostic=True),
        norm_entries_read=int(values.size),
        diagnostic_entries_materialized=a.shape[0] * a.shape[1] if diagnostic_mode == 'direct' else 0,
        oracle_entries_materialized=oracle_entries,
        coefficient_mode=coefficient_mode,
        diagnostic_mode=diagnostic_mode,
        certificate_kind='energy_identity_float64' if method == 'energy' else ('gaussian_pilot_heuristic' if rho_mode == 'pilot' else 'gaussian'),
        energy_clipped=energy_clipped,
        costs_json=json.dumps({'counts': costs.counts, 'seconds': costs.seconds}, sort_keys=True),
        schedule=schedule,
    )


def matrix_files(data_dir: Path) -> list[Path]:
    suffixes = {".npy", ".npz", ".mtx"}
    data_dir.mkdir(parents=True, exist_ok=True)
    return sorted(path for path in data_dir.iterdir() if path.suffix.lower() in suffixes)


def filter_matrix_files(files: list[Path], include: list[str] | None, exclude: list[str] | None) -> list[Path]:
    include_set = set(include or [])
    exclude_set = set(exclude or [])
    filtered = files
    if include_set:
        filtered = [path for path in filtered if path.stem in include_set]
    if exclude_set:
        filtered = [path for path in filtered if path.stem not in exclude_set]
    return filtered


def empty_data_message(data_dir: Path) -> str:
    return (
        f"no matrices found in {data_dir}\n\n"
        "Place one or more matrix files in that directory and rerun the script.\n"
        "Supported formats: .npy, .npz, .mtx\n\n"
        "Examples:\n"
        "  python run_real_matrices.py --data-dir data\n"
        "  python experiments/real_matrix_study/run_real_matrices.py "
        "--data-dir experiments/real_matrix_study/data"
    )


def write_trials(path: Path, rows: list[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(TrialResult.__dataclass_fields__.keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def summarize(rows: list[TrialResult]) -> list[dict[str, float | str]]:
    summary = []
    keys = sorted({(row.matrix, row.method) for row in rows})
    for matrix, method in keys:
        group = [row for row in rows if row.matrix == matrix and row.method == method]
        first = group[0]
        summary.append(
            {
                "matrix": matrix,
                "rows": first.rows,
                "cols": first.cols,
                "nnz": first.nnz,
                "sparse": first.sparse,
                "method": method,
                "trials": len(group),
                "mean_rank": sum(row.rank for row in group) / len(group),
                "mean_inspections": sum(row.inspections for row in group) / len(group),
                "mean_residual_ratio": sum(row.residual_ratio for row in group) / len(group),
                "mean_certificate_ratio": sum(row.certificate_ratio for row in group) / len(group),
                "mean_runtime_sec": sum(row.runtime_sec for row in group) / len(group),
                "mean_qb_runtime_sec": sum(row.qb_runtime_sec for row in group) / len(group),
                "mean_certification_runtime_sec": sum(row.certification_runtime_sec for row in group) / len(group),
                "mean_residual_eval_runtime_sec": sum(row.residual_eval_runtime_sec for row in group) / len(group),
                "mean_qb_matvecs": sum(row.qb_matvecs for row in group) / len(group),
                "mean_certification_matvecs": sum(row.certification_matvecs for row in group) / len(group),
                "mean_residual_eval_matvecs": sum(row.residual_eval_matvecs for row in group) / len(group),
                "mean_total_matvecs": sum(row.total_matvecs for row in group) / len(group),
                "false_stop_rate": sum(row.false_stop for row in group) / len(group),
                "certified_stop_rate": sum(row.certified for row in group) / len(group),
            }
        )
        for field in ('solver_runtime_sec', 'b_runtime_sec', 'pilot_runtime_sec', 'oracle_runtime_sec',
                      'norm_runtime_sec', 'solver_A', 'solver_AT', 'diagnostic_A', 'diagnostic_AT',
                      'norm_entries_read', 'diagnostic_entries_materialized', 'oracle_entries_materialized'):
            summary[-1]['mean_' + field] = sum(getattr(row, field) for row in group) / len(group)
        ledgers = [json.loads(row.costs_json)['counts'] for row in group]
        for phase in PHASES:
            for direction in ('A', 'AT'):
                summary[-1][f'mean_{phase}_{direction}'] = sum(ledger[phase][direction] for ledger in ledgers) / len(group)
    return summary


def write_dicts(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overhead_rows(summary: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    rows = []
    for matrix in sorted({str(row["matrix"]) for row in summary}):
        naive = next(row for row in summary if row["matrix"] == matrix and row["method"] == "naive")
        anytime = next(row for row in summary if row["matrix"] == matrix and row["method"] == "anytime")
        rows.append(
            {
                "matrix": matrix,
                "rows": int(naive["rows"]),
                "cols": int(naive["cols"]),
                "nnz": int(naive["nnz"]),
                "sparse": int(naive["sparse"]),
                "naive_rank": float(naive["mean_rank"]),
                "anytime_rank": float(anytime["mean_rank"]),
                "rank_overhead": float(anytime["mean_rank"]) - float(naive["mean_rank"]),
                "naive_residual_ratio": float(naive["mean_residual_ratio"]),
                "anytime_residual_ratio": float(anytime["mean_residual_ratio"]),
                "naive_certificate_ratio": float(naive["mean_certificate_ratio"]),
                "anytime_certificate_ratio": float(anytime["mean_certificate_ratio"]),
                "naive_runtime_sec": float(naive["mean_solver_runtime_sec"]),
                "anytime_runtime_sec": float(anytime["mean_solver_runtime_sec"]),
                "runtime_overhead_sec": float(anytime["mean_solver_runtime_sec"]) - float(naive["mean_solver_runtime_sec"]),
                "naive_qb_runtime_sec": float(naive["mean_qb_runtime_sec"]),
                "anytime_qb_runtime_sec": float(anytime["mean_qb_runtime_sec"]),
                "naive_certification_runtime_sec": float(naive["mean_certification_runtime_sec"]),
                "anytime_certification_runtime_sec": float(anytime["mean_certification_runtime_sec"]),
                "naive_matvecs": float(naive["mean_total_matvecs"]),
                "anytime_matvecs": float(anytime["mean_total_matvecs"]),
                "matvec_overhead": float(anytime["mean_total_matvecs"]) - float(naive["mean_total_matvecs"]),
                "anytime_certification_matvecs": float(anytime["mean_certification_matvecs"]),
                "anytime_failure_rate": float(anytime["false_stop_rate"]),
                "anytime_certified_stop_rate": float(anytime["certified_stop_rate"]),
            }
        )
        for label, source in (('naive', naive), ('anytime', anytime)):
            for direction in ('A', 'AT'):
                rows[-1][f'{label}_solver_{direction}'] = source[f'mean_solver_{direction}']
    return rows


def display_matrix_name(name: str) -> str:
    names = {
        "camera_like_192x192": "Camera-like",
        "digits_1797x64": "Digits",
        "elliptic_laplacian_289x289": "Laplacian",
        "SNAP_ca-HepPh": "ca-HepPh",
        "SNAP_email-Enron": "email-Enron",
        "SNAP_web-NotreDame": "web-NotreDame",
        "skimage_astronaut_gray_384x384": "Astronaut",
        "skimage_camera_512x512": "Camera",
        "skimage_coins_303x384": "Coins",
        "term_document_260x190": "Term-doc",
    }
    return names.get(name, name.replace("_", "\\_"))


def write_latex_table(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Real matrix adaptive QB results. The residual column reports the final residual-to-tolerance ratio $\\|A-QB\\|_{\\mathrm{F}}/\\varepsilon$ for the anytime run, so values below one meet the tolerance.}",
        "  \\label{tab:real-matrices}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\begin{tabular}{lrrrrl}",
        "    \\toprule",
        "    Matrix & Size & Naive $k$ & Anytime $k$ & Residual & Status \\\\",
        "    \\midrule",
    ]
    for row in rows:
        size = f"${int(row['rows'])}\\times {int(row['cols'])}$"
        rate = float(row["anytime_certified_stop_rate"])
        status = 'certified' if rate == 1 else ('cap reached' if rate == 0 else 'mixed')
        lines.append(
            "    "
            f"{display_matrix_name(str(row['matrix']))} & "
            f"{size} & "
            f"{float(row['naive_rank']):.1f} & "
            f"{float(row['anytime_rank']):.1f} & "
            f"{float(row['anytime_residual_ratio']):.3f} & "
            f"{status} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def write_cost_table(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{R1 solver cost including input norm, pilot/oracle work, and output coefficient construction, but excluding offline residual diagnostics. Times are seconds; $N_A$ and $N_{A^\\top}$ count right-hand sides, not block calls. Stored-entry norm scans and oracle materialization are recorded separately in the CSV files.}",
        "  \\label{tab:real-matrix-costs}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{lrrrrrr}",
        "    \\toprule",
        "    Matrix & Naive s & Anytime s & Naive $N_A$ & Naive $N_{A^\\top}$ & Anytime $N_A$ & Anytime $N_{A^\\top}$ \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            f"{display_matrix_name(str(row['matrix']))} & "
            f"{float(row['naive_runtime_sec']):.2f} & "
            f"{float(row['anytime_runtime_sec']):.2f} & "
            f"{float(row['naive_solver_A']):.0f} & "
            f"{float(row['naive_solver_AT']):.0f} & "
            f"{float(row['anytime_solver_A']):.0f} & "
            f"{float(row['anytime_solver_AT']):.0f} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/real_matrix_study/data"))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--cert-size", type=int, default=80)
    parser.add_argument("--cert-batch-size", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rel-tol", type=float, default=0.1)
    parser.add_argument("--max-rank", type=int, default=150)
    parser.add_argument("--rho-mode", choices=["oracle", "conservative", "pilot"], default="oracle")
    parser.add_argument("--rho-pilot-size", type=int, default=20)
    parser.add_argument("--rho-pilot-shrink", type=float, default=0.25)
    parser.add_argument("--include", nargs="*", default=None, help="optional matrix stems to include")
    parser.add_argument("--exclude", nargs="*", default=None, help="optional matrix stems to exclude")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--adaptive-c", type=float, default=None)
    parser.add_argument('--coefficient-mode', choices=['final', 'incremental'], default='final')
    parser.add_argument('--diagnostic-mode', choices=['energy', 'direct'], default='energy')
    args = parser.parse_args()

    files = filter_matrix_files(matrix_files(args.data_dir), args.include, args.exclude)
    if not files:
        raise SystemExit(empty_data_message(args.data_dir))

    rng = np.random.default_rng(args.seed)
    rows: list[TrialResult] = []
    for path in files:
        a, info = load_matrix(path)
        print(f"loaded {info.name}: {info.rows}x{info.cols}, sparse={info.sparse}, nnz={info.nnz}", flush=True)
        for trial in range(args.trials):
            for method in ["naive", "anytime"]:
                print(f"  trial {trial + 1}/{args.trials}, method={method}", flush=True)
                rows.append(
                    run_one(
                        rng=rng,
                        a=a,
                        info=info,
                        method=method,
                        trial=trial,
                        block_size=args.block_size,
                        cert_size=args.cert_size,
                        alpha=args.alpha,
                        rel_tol=args.rel_tol,
                        max_rank=args.max_rank,
                        rho_mode=args.rho_mode,
                        cert_batch_size=args.cert_batch_size,
                        rho_pilot_size=args.rho_pilot_size,
                        rho_pilot_shrink=args.rho_pilot_shrink,
                        adaptive_c=args.adaptive_c,
                        coefficient_mode=args.coefficient_mode,
                        diagnostic_mode=args.diagnostic_mode,
                    )
                )

    base = Path(__file__).resolve().parent
    results_dir = args.results_dir if args.results_dir is not None else base / "revision_R1" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    configuration = dict(vars(args), python=platform.python_version(), numpy=np.__version__,
                         scipy=scipy.__version__ if scipy is not None else None)
    (results_dir / 'config.json').write_text(json.dumps(configuration, indent=2, default=str) + '\n')
    summary = summarize(rows)
    overhead = overhead_rows(summary)
    write_trials(results_dir / "real_matrix_trials.csv", rows)
    write_dicts(results_dir / "real_matrix_summary.csv", summary)
    write_dicts(results_dir / "real_matrix_overhead.csv", overhead)
    write_latex_table(results_dir / "real_matrix_table.tex", overhead)
    write_cost_table(results_dir / "real_matrix_cost_table.tex", overhead)

    print(f"wrote: {results_dir / 'real_matrix_trials.csv'}")
    print(f"wrote: {results_dir / 'real_matrix_summary.csv'}")
    print(f"wrote: {results_dir / 'real_matrix_overhead.csv'}")
    print(f"wrote: {results_dir / 'real_matrix_table.tex'}")
    print(f"wrote: {results_dir / 'real_matrix_cost_table.tex'}")


if __name__ == "__main__":
    main()

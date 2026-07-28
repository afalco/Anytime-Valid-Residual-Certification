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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
        a = a.astype(float)
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


def residual_product(a: Any, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    y = matmul(a, omega)
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
) -> float:
    total = 0.0
    remaining = cert_size
    while remaining > 0:
        batch = min(remaining, cert_batch_size)
        omega = rng.standard_normal((a.shape[1], batch))
        sketches = residual_product(a, q, omega)
        total += float(np.sum(sketches * sketches))
        remaining -= batch
    z = total / cert_size
    inflation = 2.0 * math.sqrt(math.log(1.0 / alpha_level) / (cert_size * rho))
    if inflation >= 1.0:
        return math.inf
    return z / (1.0 - inflation)


def effective_rank_pilot(
    rng: np.random.Generator,
    a: Any,
    q: np.ndarray,
    pilot_size: int,
    shrink: float,
) -> float:
    omega = rng.standard_normal((a.shape[1], pilot_size))
    residual_sketch = residual_product(a, q, omega)
    trace_estimate = float(np.sum(residual_sketch * residual_sketch) / pilot_size)
    if trace_estimate <= 0.0:
        return 1.0
    normal_sketch = matmul(a.T, residual_sketch)
    fourth_estimate = float(np.sum(normal_sketch * normal_sketch) / pilot_size)
    if fourth_estimate <= 0.0:
        return 1.0
    return max(1.0, shrink * trace_estimate * trace_estimate / fourth_estimate)


def qb_step(rng: np.random.Generator, a: Any, q: np.ndarray, block_size: int) -> np.ndarray:
    omega = rng.standard_normal((a.shape[1], block_size))
    y = residual_product(a, q, omega)
    q_new, _ = np.linalg.qr(y)
    if q.size:
        q_new = q_new - q @ (q.T @ q_new)
        q_new, _ = np.linalg.qr(q_new)
        q_all, _ = np.linalg.qr(np.column_stack([q, q_new]))
        return q_all
    return q_new


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
) -> TrialResult:
    start = time.perf_counter()
    q = np.empty((a.shape[0], 0))
    norm_a = fro_norm(a)
    tolerance = rel_tol * norm_a
    inspection = 0
    certificate = math.inf
    qb_runtime = 0.0
    certification_runtime = 0.0
    rho_runtime = 0.0
    qb_matvecs = 0
    certification_matvecs = 0
    rho_pilot_matvecs = 0

    while q.shape[1] < min(max_rank, a.shape[0], a.shape[1]):
        step_start = time.perf_counter()
        q = qb_step(rng, a, q, block_size)
        qb_runtime += time.perf_counter() - step_start
        qb_matvecs += block_size
        inspection += 1

        rho_start = time.perf_counter()
        if rho_mode == "conservative":
            rho = 1.0
        elif rho_mode == "oracle":
            rho = max(1.0, effective_rank_oracle(a, q))
        else:
            rho = effective_rank_pilot(rng, a, q, rho_pilot_size, rho_pilot_shrink)
            rho_pilot_matvecs += 2 * rho_pilot_size
        rho_runtime += time.perf_counter() - rho_start

        alpha_level = alpha if method == "naive" else alpha_spending(alpha, inspection)
        cert_start = time.perf_counter()
        certificate = residual_certificate(rng, a, q, alpha_level, cert_size, rho, cert_batch_size)
        certification_runtime += time.perf_counter() - cert_start
        certification_matvecs += cert_size
        if certificate <= tolerance * tolerance:
            break

    residual_start = time.perf_counter()
    residual = residual_fro_exact(a, q, norm_a)
    residual_eval_runtime = time.perf_counter() - residual_start
    residual_eval_matvecs = int(q.shape[1])
    elapsed = time.perf_counter() - start
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
        residual_ratio=residual / tolerance if tolerance > 0 else math.inf,
        certificate_ratio=math.sqrt(certificate) / tolerance if math.isfinite(certificate) and tolerance > 0 else math.inf,
        runtime_sec=elapsed,
        qb_runtime_sec=qb_runtime,
        certification_runtime_sec=certification_runtime + rho_runtime,
        residual_eval_runtime_sec=residual_eval_runtime,
        qb_matvecs=qb_matvecs,
        certification_matvecs=certification_matvecs + rho_pilot_matvecs,
        residual_eval_matvecs=residual_eval_matvecs,
        total_matvecs=qb_matvecs + certification_matvecs + rho_pilot_matvecs + residual_eval_matvecs,
        false_stop=int(residual > tolerance),
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
            }
        )
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
                "naive_runtime_sec": float(naive["mean_runtime_sec"]),
                "anytime_runtime_sec": float(anytime["mean_runtime_sec"]),
                "runtime_overhead_sec": float(anytime["mean_runtime_sec"]) - float(naive["mean_runtime_sec"]),
                "naive_qb_runtime_sec": float(naive["mean_qb_runtime_sec"]),
                "anytime_qb_runtime_sec": float(anytime["mean_qb_runtime_sec"]),
                "naive_certification_runtime_sec": float(naive["mean_certification_runtime_sec"]),
                "anytime_certification_runtime_sec": float(anytime["mean_certification_runtime_sec"]),
                "naive_matvecs": float(naive["mean_total_matvecs"]),
                "anytime_matvecs": float(anytime["mean_total_matvecs"]),
                "matvec_overhead": float(anytime["mean_total_matvecs"]) - float(naive["mean_total_matvecs"]),
                "anytime_certification_matvecs": float(anytime["mean_certification_matvecs"]),
                "anytime_failure_rate": float(anytime["false_stop_rate"]),
            }
        )
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
        "  \\caption{Real matrix adaptive QB results.}",
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
        status = "certified" if float(row["anytime_certificate_ratio"]) <= 1.0 else "rank cap"
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
        "  \\caption{Computational cost of real-matrix adaptive QB experiments.}",
        "  \\label{tab:real-matrix-costs}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\begin{tabular}{lrrrrr}",
        "    \\toprule",
        "    Matrix & Naive time & Anytime time & Naive matvecs & Anytime matvecs & Cert. matvecs \\\\",
        "    \\midrule",
    ]
    for row in rows:
        lines.append(
            "    "
            f"{display_matrix_name(str(row['matrix']))} & "
            f"{float(row['naive_runtime_sec']):.2f} & "
            f"{float(row['anytime_runtime_sec']):.2f} & "
            f"{float(row['naive_matvecs']):.0f} & "
            f"{float(row['anytime_matvecs']):.0f} & "
            f"{float(row['anytime_certification_matvecs']):.0f} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}"])
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
                    )
                )

    base = Path(__file__).resolve().parent
    results_dir = args.results_dir if args.results_dir is not None else base / "results"
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

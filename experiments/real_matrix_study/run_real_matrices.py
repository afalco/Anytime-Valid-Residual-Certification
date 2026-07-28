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
) -> float:
    omega = rng.standard_normal((a.shape[1], cert_size))
    sketches = residual_product(a, q, omega)
    z = float(np.sum(sketches * sketches) / cert_size)
    inflation = 2.0 * math.sqrt(math.log(1.0 / alpha_level) / (cert_size * rho))
    if inflation >= 1.0:
        return math.inf
    return z / (1.0 - inflation)


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
) -> TrialResult:
    start = time.perf_counter()
    q = np.empty((a.shape[0], 0))
    norm_a = fro_norm(a)
    tolerance = rel_tol * norm_a
    inspection = 0
    certificate = math.inf
    residual = norm_a

    while q.shape[1] < min(max_rank, a.shape[0], a.shape[1]):
        q = qb_step(rng, a, q, block_size)
        inspection += 1
        residual = residual_fro_exact(a, q, norm_a)
        rho = 1.0 if rho_mode == "conservative" else max(1.0, effective_rank_oracle(a, q))
        alpha_level = alpha if method == "naive" else alpha_spending(alpha, inspection)
        certificate = residual_certificate(rng, a, q, alpha_level, cert_size, rho)
        if certificate <= tolerance * tolerance:
            break

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
        false_stop=int(residual > tolerance),
    )


def matrix_files(data_dir: Path) -> list[Path]:
    suffixes = {".npy", ".npz", ".mtx"}
    data_dir.mkdir(parents=True, exist_ok=True)
    return sorted(path for path in data_dir.iterdir() if path.suffix.lower() in suffixes)


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
                "naive_runtime_sec": float(naive["mean_runtime_sec"]),
                "anytime_runtime_sec": float(anytime["mean_runtime_sec"]),
                "runtime_overhead_sec": float(anytime["mean_runtime_sec"]) - float(naive["mean_runtime_sec"]),
                "anytime_failure_rate": float(anytime["false_stop_rate"]),
            }
        )
    return rows


def display_matrix_name(name: str) -> str:
    names = {
        "camera_like_192x192": "Camera-like",
        "digits_1797x64": "Digits",
        "elliptic_laplacian_289x289": "Laplacian",
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
        status = "met" if float(row["anytime_residual_ratio"]) <= 1.0 else "rank cap"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("experiments/real_matrix_study/data"))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--cert-size", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rel-tol", type=float, default=0.1)
    parser.add_argument("--max-rank", type=int, default=150)
    parser.add_argument("--rho-mode", choices=["oracle", "conservative"], default="oracle")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    files = matrix_files(args.data_dir)
    if not files:
        raise SystemExit(empty_data_message(args.data_dir))

    rng = np.random.default_rng(args.seed)
    rows: list[TrialResult] = []
    for path in files:
        a, info = load_matrix(path)
        print(f"loaded {info.name}: {info.rows}x{info.cols}, sparse={info.sparse}, nnz={info.nnz}")
        for trial in range(args.trials):
            for method in ["naive", "anytime"]:
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
                    )
                )

    base = Path(__file__).resolve().parent
    summary = summarize(rows)
    overhead = overhead_rows(summary)
    write_trials(base / "results" / "real_matrix_trials.csv", rows)
    write_dicts(base / "results" / "real_matrix_summary.csv", summary)
    write_dicts(base / "results" / "real_matrix_overhead.csv", overhead)
    write_latex_table(base / "results" / "real_matrix_table.tex", overhead)

    print(f"wrote: {base / 'results' / 'real_matrix_trials.csv'}")
    print(f"wrote: {base / 'results' / 'real_matrix_summary.csv'}")
    print(f"wrote: {base / 'results' / 'real_matrix_overhead.csv'}")
    print(f"wrote: {base / 'results' / 'real_matrix_table.tex'}")


if __name__ == "__main__":
    main()

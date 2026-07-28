"""Synthetic randomized QB experiment with naive and anytime certification."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrialResult:
    spectrum: str
    method: str
    trial: int
    rank: int
    inspections: int
    residual_fro: float
    tolerance: float
    certificate: float
    false_stop: int


def alpha_spending(alpha: float, inspection: int) -> float:
    return 6.0 * alpha / (math.pi**2 * inspection**2)


def spectrum_values(kind: str, n: int) -> np.ndarray:
    i = np.arange(1, n + 1, dtype=float)
    if kind == "exponential":
        return 0.94 ** (i - 1.0)
    if kind == "polynomial":
        return i ** -1.15
    if kind == "flat_tail":
        sigma = np.full(n, 0.035)
        sigma[:12] = 1.0
        return sigma
    if kind == "noisy_low_rank":
        sigma = np.full(n, 0.02)
        sigma[:10] = np.linspace(1.0, 0.35, 10)
        return sigma
    raise ValueError(f"unknown spectrum kind: {kind}")


def random_orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def synthetic_matrix(rng: np.random.Generator, n: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    u = random_orthogonal(rng, n)
    v = random_orthogonal(rng, n)
    sigma = spectrum_values(kind, n)
    return (u * sigma) @ v.T, sigma


def residual_matrix(a: np.ndarray, q: np.ndarray) -> np.ndarray:
    if q.size == 0:
        return a
    return a - q @ (q.T @ a)


def effective_rank(r: np.ndarray) -> float:
    singular = np.linalg.svd(r, compute_uv=False)
    energy = float(np.sum(singular**2))
    fourth = float(np.sum(singular**4))
    if energy == 0.0 or fourth == 0.0:
        return 1.0
    return (energy * energy) / fourth


def residual_certificate(
    rng: np.random.Generator,
    r: np.ndarray,
    alpha_level: float,
    cert_size: int,
    rho: float,
) -> float:
    omega = rng.standard_normal((r.shape[1], cert_size))
    sketches = r @ omega
    z = float(np.sum(sketches * sketches) / cert_size)
    inflation = 2.0 * math.sqrt(math.log(1.0 / alpha_level) / (cert_size * rho))
    if inflation >= 1.0:
        return math.inf
    return z / (1.0 - inflation)


def qb_step(rng: np.random.Generator, a: np.ndarray, q: np.ndarray, block_size: int) -> np.ndarray:
    omega = rng.standard_normal((a.shape[1], block_size))
    y = a @ omega
    if q.size:
        y = y - q @ (q.T @ y)
    q_new, _ = np.linalg.qr(y)
    if q.size:
        q_new = q_new - q @ (q.T @ q_new)
        q_new, _ = np.linalg.qr(q_new)
        q_combined, _ = np.linalg.qr(np.column_stack([q, q_new]))
        return q_combined
    return q_new


def run_one(
    rng: np.random.Generator,
    a: np.ndarray,
    spectrum: str,
    method: str,
    trial: int,
    block_size: int,
    cert_size: int,
    alpha: float,
    rel_tol: float,
    max_rank: int,
    rho_mode: str,
) -> TrialResult:
    q = np.empty((a.shape[0], 0))
    norm_a = float(np.linalg.norm(a, "fro"))
    tolerance = rel_tol * norm_a
    inspection = 0
    certificate = math.inf
    residual_norm = norm_a

    while q.shape[1] < max_rank:
        q = qb_step(rng, a, q, block_size)
        inspection += 1
        r = residual_matrix(a, q)
        residual_norm = float(np.linalg.norm(r, "fro"))
        rho = 1.0 if rho_mode == "conservative" else max(1.0, effective_rank(r))
        alpha_level = alpha if method == "naive" else alpha_spending(alpha, inspection)
        certificate = residual_certificate(rng, r, alpha_level, cert_size, rho)
        if certificate <= tolerance * tolerance:
            break

    return TrialResult(
        spectrum=spectrum,
        method=method,
        trial=trial,
        rank=int(q.shape[1]),
        inspections=inspection,
        residual_fro=residual_norm,
        tolerance=tolerance,
        certificate=certificate,
        false_stop=int(residual_norm > tolerance),
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
    keys = sorted({(r.spectrum, r.method) for r in rows})
    for spectrum, method in keys:
        group = [r for r in rows if r.spectrum == spectrum and r.method == method]
        summary.append(
            {
                "spectrum": spectrum,
                "method": method,
                "trials": len(group),
                "mean_rank": sum(r.rank for r in group) / len(group),
                "mean_inspections": sum(r.inspections for r in group) / len(group),
                "false_stop_rate": sum(r.false_stop for r in group) / len(group),
                "mean_residual_ratio": sum(r.residual_fro / r.tolerance for r in group) / len(group),
            }
        )
    return summary


def write_summary(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "spectrum",
        "method",
        "trials",
        "mean_rank",
        "mean_inspections",
        "false_stop_rate",
        "mean_residual_ratio",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overhead_rows(summary: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    rows = []
    spectra = sorted({str(row["spectrum"]) for row in summary})
    for spectrum in spectra:
        naive = next(row for row in summary if row["spectrum"] == spectrum and row["method"] == "naive")
        anytime = next(row for row in summary if row["spectrum"] == spectrum and row["method"] == "anytime")
        naive_rank = float(naive["mean_rank"])
        anytime_rank = float(anytime["mean_rank"])
        naive_inspections = float(naive["mean_inspections"])
        anytime_inspections = float(anytime["mean_inspections"])
        rows.append(
            {
                "spectrum": spectrum,
                "naive_rank": naive_rank,
                "anytime_rank": anytime_rank,
                "rank_overhead": anytime_rank - naive_rank,
                "rank_overhead_percent": 100.0 * (anytime_rank - naive_rank) / naive_rank,
                "naive_inspections": naive_inspections,
                "anytime_inspections": anytime_inspections,
                "inspection_overhead": anytime_inspections - naive_inspections,
                "naive_false_stop_rate": float(naive["false_stop_rate"]),
                "anytime_false_stop_rate": float(anytime["false_stop_rate"]),
            }
        )
    return rows


def write_overhead_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "spectrum",
        "naive_rank",
        "anytime_rank",
        "rank_overhead",
        "rank_overhead_percent",
        "naive_inspections",
        "anytime_inspections",
        "inspection_overhead",
        "naive_false_stop_rate",
        "anytime_false_stop_rate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_overhead_table(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Rank and inspection overhead of anytime-valid stopping on synthetic QB experiments.}",
        "  \\label{tab:synthetic-qb-overhead}",
        "  \\begin{tabular}{lrrrr}",
        "    \\toprule",
        "    Spectrum & Naive rank & Anytime rank & Rank overhead & Inspection overhead \\\\",
        "    \\midrule",
    ]
    for row in rows:
        label = str(row["spectrum"]).replace("_", " ")
        lines.append(
            "    "
            f"{label} & "
            f"{float(row['naive_rank']):.2f} & "
            f"{float(row['anytime_rank']):.2f} & "
            f"{float(row['rank_overhead']):.2f} "
            f"({float(row['rank_overhead_percent']):.1f}\\%) & "
            f"{float(row['inspection_overhead']):.2f} \\\\"
        )
    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_rank_svg(path: Path, summary: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spectra = ["exponential", "polynomial", "flat_tail", "noisy_low_rank"]
    methods = ["naive", "anytime"]
    colors = {"naive": "#b42318", "anytime": "#155eef"}
    width, height = 920, 540
    left, right, top, bottom = 78, 36, 52, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_rank = max(float(row["mean_rank"]) for row in summary) * 1.15

    def ymap(value: float) -> float:
        return top + (1.0 - value / max_rank) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="26" font-family="Helvetica, Arial, sans-serif" font-size="18" font-weight="700">Rank selected by adaptive QB stopping rules</text>',
        f'<text x="{left}" y="46" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#475467">Naive fixed-time stopping versus anytime alpha-spending on synthetic spectra</text>',
    ]

    for tick in range(0, int(math.ceil(max_rank / 10.0)) * 10 + 1, 10):
        y = ymap(tick)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#eaecf0" stroke-width="1"/>')
        elements.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#475467">{tick}</text>')

    group_w = plot_w / len(spectra)
    bar_w = 34
    for idx, spectrum in enumerate(spectra):
        center = left + group_w * (idx + 0.5)
        for m_idx, method in enumerate(methods):
            row = next(r for r in summary if r["spectrum"] == spectrum and r["method"] == method)
            value = float(row["mean_rank"])
            x = center + (m_idx - 0.5) * (bar_w + 10)
            y = ymap(value)
            h = top + plot_h - y
            elements.append(f'<rect x="{x-bar_w/2:.2f}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" fill="{colors[method]}" opacity="0.88"/>')
            elements.append(f'<text x="{x:.2f}" y="{y-6:.2f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#344054">{value:.1f}</text>')
        label = spectrum.replace("_", " ")
        elements.append(f'<text x="{center:.2f}" y="{height-bottom+28}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#344054">{label}</text>')

    elements.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#344054" stroke-width="1.2"/>')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#344054" stroke-width="1.2"/>')
    elements.append(f'<text x="20" y="{top+plot_h/2:.2f}" transform="rotate(-90 20 {top+plot_h/2:.2f})" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#344054">Mean selected rank</text>')
    legend_x, legend_y = width - 230, top + 8
    for idx, method in enumerate(methods):
        y = legend_y + idx * 24
        elements.append(f'<rect x="{legend_x}" y="{y-12}" width="16" height="16" fill="{colors[method]}" opacity="0.88"/>')
        elements.append(f'<text x="{legend_x+26}" y="{y+1}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#344054">{method}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--cert-size", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rel-tol", type=float, default=0.1)
    parser.add_argument("--max-rank", type=int, default=100)
    parser.add_argument("--rho-mode", choices=["conservative", "oracle"], default="oracle")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    spectra = ["exponential", "polynomial", "flat_tail", "noisy_low_rank"]
    rows: list[TrialResult] = []
    for spectrum in spectra:
        for trial in range(args.trials):
            a, _ = synthetic_matrix(rng, args.n, spectrum)
            for method in ["naive", "anytime"]:
                rows.append(
                    run_one(
                        rng=rng,
                        a=a,
                        spectrum=spectrum,
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
    write_trials(base / "results" / "synthetic_qb_trials.csv", rows)
    write_summary(base / "results" / "synthetic_qb_summary.csv", summary)
    write_overhead_csv(base / "results" / "synthetic_qb_overhead.csv", overhead)
    write_latex_overhead_table(base / "results" / "synthetic_qb_overhead_table.tex", overhead)
    write_rank_svg(base / "figures" / "synthetic_qb_rank_overhead.svg", summary)

    print(f"wrote: {base / 'results' / 'synthetic_qb_trials.csv'}")
    print(f"wrote: {base / 'results' / 'synthetic_qb_summary.csv'}")
    print(f"wrote: {base / 'results' / 'synthetic_qb_overhead.csv'}")
    print(f"wrote: {base / 'results' / 'synthetic_qb_overhead_table.tex'}")
    print(f"wrote: {base / 'figures' / 'synthetic_qb_rank_overhead.svg'}")
    for row in summary:
        print(
            f"{row['spectrum']:>15} {row['method']:>7}: "
            f"rank={float(row['mean_rank']):.2f}, "
            f"false_stop={float(row['false_stop_rate']):.3f}, "
            f"res/tol={float(row['mean_residual_ratio']):.3f}"
        )


if __name__ == "__main__":
    main()

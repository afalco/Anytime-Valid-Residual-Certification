"""Sensitivity of anytime QB stopping to the effective-rank lower bound."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from run_synthetic_qb import run_one, synthetic_matrix


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "spectrum",
        "rho_mode",
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


def write_overhead_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "spectrum",
        "oracle_rank",
        "conservative_rank",
        "rank_overhead",
        "rank_overhead_percent",
        "oracle_inspections",
        "conservative_inspections",
        "inspection_overhead",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    summary = []
    keys = sorted({(str(row["spectrum"]), str(row["rho_mode"])) for row in rows})
    for spectrum, rho_mode in keys:
        group = [row for row in rows if row["spectrum"] == spectrum and row["rho_mode"] == rho_mode]
        summary.append(
            {
                "spectrum": spectrum,
                "rho_mode": rho_mode,
                "trials": len(group),
                "mean_rank": sum(float(row["rank"]) for row in group) / len(group),
                "mean_inspections": sum(float(row["inspections"]) for row in group) / len(group),
                "false_stop_rate": sum(float(row["false_stop"]) for row in group) / len(group),
                "mean_residual_ratio": sum(float(row["residual_ratio"]) for row in group) / len(group),
            }
        )
    return summary


def overhead(summary: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    rows = []
    for spectrum in sorted({str(row["spectrum"]) for row in summary}):
        oracle = next(row for row in summary if row["spectrum"] == spectrum and row["rho_mode"] == "oracle")
        conservative = next(row for row in summary if row["spectrum"] == spectrum and row["rho_mode"] == "conservative")
        oracle_rank = float(oracle["mean_rank"])
        conservative_rank = float(conservative["mean_rank"])
        oracle_inspections = float(oracle["mean_inspections"])
        conservative_inspections = float(conservative["mean_inspections"])
        rows.append(
            {
                "spectrum": spectrum,
                "oracle_rank": oracle_rank,
                "conservative_rank": conservative_rank,
                "rank_overhead": conservative_rank - oracle_rank,
                "rank_overhead_percent": 100.0 * (conservative_rank - oracle_rank) / oracle_rank,
                "oracle_inspections": oracle_inspections,
                "conservative_inspections": conservative_inspections,
                "inspection_overhead": conservative_inspections - oracle_inspections,
            }
        )
    return rows


def write_latex_table(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Sensitivity of anytime-valid QB stopping to effective-rank information.}",
        "  \\label{tab:effective-rank-sensitivity}",
        "  \\begin{tabular}{lrrrr}",
        "    \\toprule",
        "    Spectrum & Oracle rank & Conservative rank & Rank overhead & Inspection overhead \\\\",
        "    \\midrule",
    ]
    for row in rows:
        label = str(row["spectrum"]).replace("_", " ")
        lines.append(
            "    "
            f"{label} & "
            f"{float(row['oracle_rank']):.2f} & "
            f"{float(row['conservative_rank']):.2f} & "
            f"{float(row['rank_overhead']):.2f} "
            f"({float(row['rank_overhead_percent']):.1f}\\%) & "
            f"{float(row['inspection_overhead']):.2f} \\\\"
        )
    lines.extend(["    \\bottomrule", "  \\end{tabular}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def write_svg(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spectra = ["exponential", "polynomial", "flat_tail", "noisy_low_rank"]
    modes = ["oracle", "conservative"]
    colors = {"oracle": "#155eef", "conservative": "#7a271a"}
    width, height = 920, 540
    left, right, top, bottom = 78, 36, 52, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_rank = max(float(row[f"{mode}_rank"]) for row in rows for mode in modes) * 1.15

    def ymap(value: float) -> float:
        return top + (1.0 - value / max_rank) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="26" font-family="Helvetica, Arial, sans-serif" font-size="18" font-weight="700">Effective-rank information reduces certification overhead</text>',
        f'<text x="{left}" y="46" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#475467">Anytime stopping with oracle lower bound versus conservative rho = 1</text>',
    ]
    for tick in range(0, int(math.ceil(max_rank / 10.0)) * 10 + 1, 10):
        y = ymap(tick)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#eaecf0" stroke-width="1"/>')
        elements.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#475467">{tick}</text>')

    group_w = plot_w / len(spectra)
    bar_w = 34
    for idx, spectrum in enumerate(spectra):
        center = left + group_w * (idx + 0.5)
        row = next(r for r in rows if r["spectrum"] == spectrum)
        for m_idx, mode in enumerate(modes):
            value = float(row[f"{mode}_rank"])
            x = center + (m_idx - 0.5) * (bar_w + 10)
            y = ymap(value)
            h = top + plot_h - y
            elements.append(f'<rect x="{x-bar_w/2:.2f}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" fill="{colors[mode]}" opacity="0.88"/>')
            elements.append(f'<text x="{x:.2f}" y="{y-6:.2f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#344054">{value:.1f}</text>')
        label = spectrum.replace("_", " ")
        elements.append(f'<text x="{center:.2f}" y="{height-bottom+28}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#344054">{label}</text>')

    elements.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#344054" stroke-width="1.2"/>')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#344054" stroke-width="1.2"/>')
    elements.append(f'<text x="20" y="{top+plot_h/2:.2f}" transform="rotate(-90 20 {top+plot_h/2:.2f})" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#344054">Mean selected rank</text>')
    legend_x, legend_y = width - 245, top + 8
    for idx, mode in enumerate(modes):
        y = legend_y + idx * 24
        elements.append(f'<rect x="{legend_x}" y="{y-12}" width="16" height="16" fill="{colors[mode]}" opacity="0.88"/>')
        elements.append(f'<text x="{legend_x+26}" y="{y+1}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#344054">{mode}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--cert-size", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rel-tol", type=float, default=0.1)
    parser.add_argument("--max-rank", type=int, default=110)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    spectra = ["exponential", "polynomial", "flat_tail", "noisy_low_rank"]
    trial_rows: list[dict[str, float | str]] = []
    for spectrum in spectra:
        for trial in range(args.trials):
            a, _ = synthetic_matrix(rng, args.n, spectrum)
            for rho_mode in ["oracle", "conservative"]:
                result = run_one(
                    rng=rng,
                    a=a,
                    spectrum=spectrum,
                    method="anytime",
                    trial=trial,
                    block_size=args.block_size,
                    cert_size=args.cert_size,
                    alpha=args.alpha,
                    rel_tol=args.rel_tol,
                    max_rank=args.max_rank,
                    rho_mode=rho_mode,
                )
                trial_rows.append(
                    {
                        "spectrum": spectrum,
                        "rho_mode": rho_mode,
                        "trial": trial,
                        "rank": result.rank,
                        "inspections": result.inspections,
                        "false_stop": result.false_stop,
                        "residual_ratio": result.residual_fro / result.tolerance,
                    }
                )

    base = Path(__file__).resolve().parent
    summary = summarize(trial_rows)
    overhead_rows = overhead(summary)
    write_csv(base / "results" / "effective_rank_sensitivity_summary.csv", summary)
    write_overhead_csv(base / "results" / "effective_rank_sensitivity_overhead.csv", overhead_rows)
    write_latex_table(base / "results" / "effective_rank_sensitivity_table.tex", overhead_rows)
    write_svg(base / "figures" / "effective_rank_sensitivity.svg", overhead_rows)

    print(f"wrote: {base / 'results' / 'effective_rank_sensitivity_summary.csv'}")
    print(f"wrote: {base / 'results' / 'effective_rank_sensitivity_overhead.csv'}")
    print(f"wrote: {base / 'results' / 'effective_rank_sensitivity_table.tex'}")
    print(f"wrote: {base / 'figures' / 'effective_rank_sensitivity.svg'}")
    for row in overhead_rows:
        print(
            f"{row['spectrum']:>15}: "
            f"oracle={float(row['oracle_rank']):.2f}, "
            f"conservative={float(row['conservative_rank']):.2f}, "
            f"overhead={float(row['rank_overhead']):.2f}"
        )


if __name__ == "__main__":
    main()

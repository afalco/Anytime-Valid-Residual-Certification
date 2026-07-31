"""Direct coverage diagnostics for adaptive randomized QB stopping."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from run_synthetic_qb import (
    TrialResult,
    run_one,
    synthetic_matrix,
)


def summarize(rows: list[TrialResult], alpha: float) -> list[dict[str, float | str]]:
    summary = []
    keys = sorted({(row.spectrum, row.method) for row in rows})
    for spectrum, method in keys:
        group = [row for row in rows if row.spectrum == spectrum and row.method == method]
        summary.append(
            {
                "spectrum": spectrum,
                "method": method,
                "alpha": alpha,
                "trials": len(group),
                "mean_rank": sum(row.rank for row in group) / len(group),
                "mean_inspections": sum(row.inspections for row in group) / len(group),
                "certificate_miscoverage_rate": sum(
                    row.certificate < row.residual_fro * row.residual_fro for row in group
                )
                / len(group),
                "false_stop_rate": sum(row.false_stop for row in group) / len(group),
                "mean_residual_to_tolerance": sum(row.residual_fro / row.tolerance for row in group)
                / len(group),
            }
        )
    return summary


def write_dicts(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    spectra = ["exponential", "flat_tail", "noisy_low_rank", "polynomial"]
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Direct adaptive-QB coverage diagnostic. The table reports empirical failure rates at the data-dependent stopping rank over synthetic matrices. Certificate miscoverage means that the final upper certificate is below the exact squared residual; false stop means \\(\\|R_\\tau\\|_{\\mathrm{F}}>\\varepsilon\\) at a certified stop. The diagnostic uses oracle effective-rank information in the fixed-time certificates, so it isolates the effect of repeated inspection rather than conservatism from \\(\\rho=1\\).}",
        "  \\label{tab:adaptive-qb-coverage}",
        "  \\small",
        "  \\setlength{\\tabcolsep}{4pt}",
        "  \\resizebox{\\textwidth}{!}{%",
        "  \\begin{tabular}{llrrrrr}",
        "    \\toprule",
        "    Spectrum & Method & Mean rank & Inspections & Cert. miscoverage & False stop & Residual/tol. \\\\",
        "    \\midrule",
    ]
    for spectrum in spectra:
        for method in ["naive", "anytime"]:
            row = next(item for item in rows if item["spectrum"] == spectrum and item["method"] == method)
            lines.append(
                "    "
                f"{spectrum.replace('_', ' ')} & "
                f"{method} & "
                f"{float(row['mean_rank']):.1f} & "
                f"{float(row['mean_inspections']):.1f} & "
                f"{float(row['certificate_miscoverage_rate']):.3f} & "
                f"{float(row['false_stop_rate']):.3f} & "
                f"{float(row['mean_residual_to_tolerance']):.3f} \\\\"
            )
    lines.extend(["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--block-size", type=int, default=1)
    parser.add_argument("--cert-size", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--rel-tol", type=float, default=0.1)
    parser.add_argument("--max-rank", type=int, default=80)
    parser.add_argument("--seed", type=int, default=127)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows: list[TrialResult] = []
    for spectrum in ["exponential", "polynomial", "flat_tail", "noisy_low_rank"]:
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
                        rho_mode="oracle",
                    )
                )

    base = Path(__file__).resolve().parent
    summary = summarize(rows, args.alpha)
    write_dicts(base / "results" / "adaptive_qb_coverage_summary.csv", summary)
    write_latex_table(base / "results" / "adaptive_qb_coverage_table.tex", summary)

    print(f"wrote: {base / 'results' / 'adaptive_qb_coverage_summary.csv'}")
    print(f"wrote: {base / 'results' / 'adaptive_qb_coverage_table.tex'}")
    for row in summary:
        print(
            f"{row['spectrum']:>15} {row['method']:>7}: "
            f"cert_miscoverage={float(row['certificate_miscoverage_rate']):.3f}, "
            f"false_stop={float(row['false_stop_rate']):.3f}, "
            f"rank={float(row['mean_rank']):.1f}"
        )


if __name__ == "__main__":
    main()

"""Regenerate all numerical experiments for the paper."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def convert_svg(svg_path: Path, pdf_path: Path) -> None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        print(f"skipping PDF conversion; rsvg-convert not found: {svg_path}")
        return
    run([converter, "-f", "pdf", "-o", str(pdf_path), str(svg_path)])


def main() -> None:
    run(
        [
            PYTHON,
            "experiments/miscoverage_study/run_miscoverage.py",
            "--alpha",
            "0.05",
            "--max-checks",
            "100",
            "--trials",
            "20000",
            "--seed",
            "0",
        ]
    )
    convert_svg(
        ROOT / "experiments/miscoverage_study/figures/miscoverage_optional_stopping.svg",
        ROOT / "experiments/miscoverage_study/figures/miscoverage_optional_stopping.pdf",
    )

    run([PYTHON, "experiments/synthetic_qb_study/run_synthetic_qb.py"])
    convert_svg(
        ROOT / "experiments/synthetic_qb_study/figures/synthetic_qb_rank_overhead.svg",
        ROOT / "experiments/synthetic_qb_study/figures/synthetic_qb_rank_overhead.pdf",
    )

    run([PYTHON, "experiments/synthetic_qb_study/run_effective_rank_sensitivity.py"])
    convert_svg(
        ROOT / "experiments/synthetic_qb_study/figures/effective_rank_sensitivity.svg",
        ROOT / "experiments/synthetic_qb_study/figures/effective_rank_sensitivity.pdf",
    )

    data_dir = ROOT / "experiments/real_matrix_study/data"
    run([PYTHON, "experiments/real_matrix_study/generate_benchmark_matrices.py", "--out-dir", str(data_dir)])
    run([PYTHON, "experiments/real_matrix_study/generate_application_matrices.py", "--out-dir", str(data_dir)])
    run(
        [
            PYTHON,
            "experiments/real_matrix_study/run_real_matrices.py",
            "--data-dir",
            str(data_dir),
            "--trials",
            "5",
            "--block-size",
            "5",
            "--cert-size",
            "80",
            "--alpha",
            "0.05",
            "--rel-tol",
            "0.1",
            "--max-rank",
            "150",
            "--rho-mode",
            "oracle",
        ]
    )
    run(
        [
            PYTHON,
            "experiments/real_matrix_study/plot_real_matrix_ranks.py",
            "--results",
            "experiments/real_matrix_study/results/real_matrix_overhead.csv",
            "--out-dir",
            "experiments/real_matrix_study/figures",
        ]
    )
    print("All experiment artifacts regenerated.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)

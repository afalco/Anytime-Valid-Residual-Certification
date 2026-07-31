"""Run all experiments used by the current paper draft.

This script regenerates numerical CSV files, SVG/PDF figures, and manuscript
copies for the experiments currently included in paper.tex.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/Users/antoniofalco/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3")


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def convert_svg(svg_path: Path, pdf_path: Path) -> None:
    converter = shutil.which("rsvg-convert")
    if converter is None:
        raise RuntimeError("rsvg-convert is required to produce manuscript-ready PDF figures")
    run([converter, "-f", "pdf", "-o", str(pdf_path), str(svg_path)])


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied: {src} -> {dst}")


def main() -> None:
    run(
        [
            str(PYTHON),
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
    copy(
        ROOT / "experiments/miscoverage_study/figures/miscoverage_optional_stopping.pdf",
        ROOT / "manuscript/figures/miscoverage_optional_stopping.pdf",
    )

    run([str(PYTHON), "experiments/synthetic_qb_study/run_synthetic_qb.py"])
    convert_svg(
        ROOT / "experiments/synthetic_qb_study/figures/synthetic_qb_rank_overhead.svg",
        ROOT / "experiments/synthetic_qb_study/figures/synthetic_qb_rank_overhead.pdf",
    )
    copy(
        ROOT / "experiments/synthetic_qb_study/figures/synthetic_qb_rank_overhead.pdf",
        ROOT / "manuscript/figures/synthetic_qb_rank_overhead.pdf",
    )
    copy(
        ROOT / "experiments/synthetic_qb_study/results/synthetic_qb_overhead_table.tex",
        ROOT / "manuscript/tables/synthetic_qb_overhead_table.tex",
    )

    run(
        [
            str(PYTHON),
            "experiments/synthetic_qb_study/run_adaptive_qb_coverage.py",
            "--trials",
            "500",
            "--n",
            "80",
            "--block-size",
            "1",
            "--cert-size",
            "3",
            "--alpha",
            "0.25",
            "--rel-tol",
            "0.1",
            "--max-rank",
            "80",
            "--seed",
            "127",
        ]
    )
    copy(
        ROOT / "experiments/synthetic_qb_study/results/adaptive_qb_coverage_table.tex",
        ROOT / "manuscript/tables/adaptive_qb_coverage_table.tex",
    )

    run([str(PYTHON), "experiments/synthetic_qb_study/run_effective_rank_sensitivity.py"])
    convert_svg(
        ROOT / "experiments/synthetic_qb_study/figures/effective_rank_sensitivity.svg",
        ROOT / "experiments/synthetic_qb_study/figures/effective_rank_sensitivity.pdf",
    )
    copy(
        ROOT / "experiments/synthetic_qb_study/figures/effective_rank_sensitivity.pdf",
        ROOT / "manuscript/figures/effective_rank_sensitivity.pdf",
    )
    copy(
        ROOT / "experiments/synthetic_qb_study/results/effective_rank_sensitivity_table.tex",
        ROOT / "manuscript/tables/effective_rank_sensitivity_table.tex",
    )

    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"])
    print("All experiments and manuscript artifacts regenerated.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)

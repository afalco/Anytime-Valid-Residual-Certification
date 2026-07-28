"""Plot selected-rank comparisons for the real-matrix study."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="fontconfig-"))

import matplotlib.pyplot as plt
import numpy as np


DISPLAY_NAMES = {
    "camera_like_192x192": "Camera-like",
    "digits_1797x64": "Digits",
    "elliptic_laplacian_289x289": "Laplacian",
    "skimage_astronaut_gray_384x384": "Astronaut",
    "skimage_camera_512x512": "Camera",
    "skimage_coins_303x384": "Coins",
    "term_document_260x190": "Term-doc",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def display_name(name: str) -> str:
    return DISPLAY_NAMES.get(name, name.replace("_", " "))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/real_matrix_overhead.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    rows = read_rows(args.results)
    if not rows:
        raise SystemExit(f"no rows found in {args.results}")

    labels = [display_name(row["matrix"]) for row in rows]
    naive = np.array([float(row["naive_rank"]) for row in rows])
    anytime = np.array([float(row["anytime_rank"]) for row in rows])
    residual = np.array([float(row["anytime_residual_ratio"]) for row in rows])

    x = np.arange(len(rows))
    width = 0.38

    fig_width = max(7.0, 0.95 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    ax.bar(x - width / 2, naive, width, label="Naive fixed-time", color="#4c78a8")
    ax.bar(x + width / 2, anytime, width, label="Anytime alpha-spending", color="#f58518")

    for xpos, rank, ratio in zip(x + width / 2, anytime, residual, strict=True):
        if ratio > 1.0:
            ax.text(xpos, rank + 2.0, "cap", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Mean selected rank")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="upper left", handletextpad=1.8)
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg"]:
        target = args.out_dir / f"real_matrix_rank_comparison.{suffix}"
        fig.savefig(target)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()

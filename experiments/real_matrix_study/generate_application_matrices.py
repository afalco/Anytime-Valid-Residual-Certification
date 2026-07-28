"""Generate reproducible application matrices for the real-matrix study.

The generated matrices come from scikit-learn and scikit-image example
datasets. They are small enough for repeated QB trials, but they are not
diagonal synthetic spectra.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from skimage import color, data, transform
from sklearn.datasets import load_digits


def normalize_unit_interval(a: np.ndarray) -> np.ndarray:
    values = np.asarray(a, dtype=float)
    values = values - float(np.min(values))
    scale = float(np.max(values))
    if scale > 0.0:
        values = values / scale
    return values


def save_matrix(path: Path, matrix: np.ndarray, description: str) -> dict[str, str | int]:
    matrix = np.asarray(matrix, dtype=float)
    np.save(path, matrix)
    return {
        "file": path.name,
        "rows": matrix.shape[0],
        "cols": matrix.shape[1],
        "source": description,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []

    digits = load_digits().data.astype(float)
    digits = digits - digits.mean(axis=0, keepdims=True)
    rows.append(save_matrix(args.out_dir / "digits_1797x64.npy", digits, "sklearn.load_digits centered samples"))

    rows.append(
        save_matrix(
            args.out_dir / "skimage_camera_512x512.npy",
            normalize_unit_interval(data.camera()),
            "skimage.data.camera grayscale image",
        )
    )

    rows.append(
        save_matrix(
            args.out_dir / "skimage_coins_303x384.npy",
            normalize_unit_interval(data.coins()),
            "skimage.data.coins grayscale image",
        )
    )

    astronaut = color.rgb2gray(data.astronaut())
    astronaut = transform.resize(astronaut, (384, 384), anti_aliasing=True)
    rows.append(
        save_matrix(
            args.out_dir / "skimage_astronaut_gray_384x384.npy",
            normalize_unit_interval(astronaut),
            "skimage.data.astronaut grayscale resized image",
        )
    )

    manifest = args.out_dir / "application_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "rows", "cols", "source"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} application matrices to {args.out_dir}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()

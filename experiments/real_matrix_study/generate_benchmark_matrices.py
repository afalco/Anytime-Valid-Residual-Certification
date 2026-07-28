"""Generate a small local matrix pack for the real-matrix QB study.

The matrices created here are reproducible benchmark matrices meant to exercise
the experimental pipeline before importing external collections such as
SuiteSparse. They are not synthetic singular-value diagonal matrices.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def camera_like_matrix(rng: np.random.Generator, n: int = 192) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, n)
    x, y = np.meshgrid(grid, grid, indexing="ij")
    image = (
        0.55 * np.exp(-35.0 * ((x - 0.28) ** 2 + (y - 0.72) ** 2))
        + 0.35 * np.exp(-55.0 * ((x - 0.72) ** 2 + (y - 0.38) ** 2))
        + 0.20 * np.sin(5.0 * np.pi * x) * np.cos(3.0 * np.pi * y)
        + 0.15 * (x > 0.58)
        + 0.08 * y
    )
    image += 0.015 * rng.standard_normal(image.shape)
    image -= image.min()
    image /= image.max()
    return image


def term_document_matrix(
    rng: np.random.Generator,
    terms: int = 260,
    documents: int = 190,
    topics: int = 9,
) -> sp.csr_matrix:
    topic_words = rng.gamma(shape=0.6, scale=1.0, size=(terms, topics))
    topic_words *= rng.random((terms, topics)) < 0.22
    doc_topics = rng.gamma(shape=0.8, scale=1.0, size=(topics, documents))
    doc_topics *= rng.random((topics, documents)) < 0.55
    rates = topic_words @ doc_topics
    rates /= max(float(rates.mean()), 1e-12)
    counts = rng.poisson(0.7 * rates)
    counts[counts < 2] = 0
    return sp.csr_matrix(counts.astype(float))


def elliptic_laplacian_matrix(n_side: int = 17) -> sp.csr_matrix:
    n = n_side * n_side
    main = 4.0 * np.ones(n)
    east_west = -1.0 * np.ones(n - 1)
    north_south = -1.0 * np.ones(n - n_side)
    for idx in range(1, n):
        if idx % n_side == 0:
            east_west[idx - 1] = 0.0
    return sp.diags(
        [main, east_west, east_west, north_south, north_south],
        [0, -1, 1, -n_side, n_side],
        shape=(n, n),
        format="csr",
    )


def write_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    fields = ["file", "rows", "cols", "nnz", "format", "description"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    manifest: list[dict[str, str | int]] = []

    camera = camera_like_matrix(rng)
    camera_path = args.out_dir / "camera_like_192x192.npy"
    np.save(camera_path, camera)
    manifest.append(
        {
            "file": camera_path.name,
            "rows": camera.shape[0],
            "cols": camera.shape[1],
            "nnz": int(np.count_nonzero(camera)),
            "format": "dense npy",
            "description": "smooth image-like intensity matrix with edges and texture",
        }
    )

    term_doc = term_document_matrix(rng)
    term_doc_path = args.out_dir / "term_document_260x190.npz"
    sp.save_npz(term_doc_path, term_doc)
    manifest.append(
        {
            "file": term_doc_path.name,
            "rows": term_doc.shape[0],
            "cols": term_doc.shape[1],
            "nnz": int(term_doc.nnz),
            "format": "sparse npz",
            "description": "topic-model term-document count matrix",
        }
    )

    laplacian = elliptic_laplacian_matrix()
    laplacian_path = args.out_dir / "elliptic_laplacian_289x289.npz"
    sp.save_npz(laplacian_path, laplacian)
    manifest.append(
        {
            "file": laplacian_path.name,
            "rows": laplacian.shape[0],
            "cols": laplacian.shape[1],
            "nnz": int(laplacian.nnz),
            "format": "sparse npz",
            "description": "five-point finite-difference elliptic operator",
        }
    )

    write_manifest(args.out_dir / "manifest.csv", manifest)
    for row in manifest:
        print(f"wrote {row['file']}: {row['rows']}x{row['cols']}, nnz={row['nnz']}")


if __name__ == "__main__":
    main()

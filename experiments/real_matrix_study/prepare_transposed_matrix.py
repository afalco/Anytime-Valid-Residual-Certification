"""Create a transposed sparse matrix file from a Matrix Market input."""

from __future__ import annotations

import argparse
from pathlib import Path

import scipy.io
import scipy.sparse as sp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    matrix = scipy.io.mmread(args.input).astype(float).tocsr()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(args.output, matrix.T.tocsr())
    print(f"wrote {args.output}: {matrix.shape} -> {matrix.T.shape}, nnz={matrix.nnz}")


if __name__ == "__main__":
    main()

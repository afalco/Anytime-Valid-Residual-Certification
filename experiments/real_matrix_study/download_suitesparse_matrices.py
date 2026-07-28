"""Download selected SuiteSparse matrices in Matrix Market format.

The default list is deliberately modest enough for repeated randomized QB
experiments on a laptop while still moving beyond toy examples.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


SUITESPARSE_MM_BASE = "https://sparse.tamu.edu/MM"


@dataclass(frozen=True)
class SuiteSparseMatrix:
    group: str
    name: str
    rows: int
    cols: int
    nnz: int
    kind: str

    @property
    def stem(self) -> str:
        return f"{self.group}_{self.name}"

    @property
    def url(self) -> str:
        return f"{SUITESPARSE_MM_BASE}/{self.group}/{self.name}.tar.gz"


DEFAULT_MATRICES = [
    SuiteSparseMatrix("SNAP", "ca-HepPh", 12008, 12008, 237010, "collaboration graph"),
    SuiteSparseMatrix("SNAP", "email-Enron", 36692, 36692, 367662, "email graph"),
    SuiteSparseMatrix("SNAP", "web-NotreDame", 325729, 325729, 1497134, "web graph"),
]


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"already downloaded: {target}")
        return
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"wrote {target}")


def extract_matrix_market(archive: Path, matrix: SuiteSparseMatrix, out_dir: Path) -> Path:
    target = out_dir / f"{matrix.stem}.mtx"
    if target.exists():
        print(f"already extracted: {target}")
        return target

    with tarfile.open(archive, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.name.endswith(".mtx")]
        preferred = [member for member in members if Path(member.name).name == f"{matrix.name}.mtx"]
        if len(preferred) != 1:
            raise RuntimeError(f"could not identify the primary .mtx file in {archive}")
        source = preferred[0]
        extracted = tar.extractfile(source)
        if extracted is None:
            raise RuntimeError(f"could not extract {source.name} from {archive}")
        with target.open("wb") as handle:
            shutil.copyfileobj(extracted, handle)
    print(f"wrote {target}")
    return target


def write_manifest(path: Path, matrices: list[SuiteSparseMatrix]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "group", "name", "rows", "cols", "nnz", "kind", "url"])
        writer.writeheader()
        for matrix in matrices:
            writer.writerow(
                {
                    "file": f"{matrix.stem}.mtx",
                    "group": matrix.group,
                    "name": matrix.name,
                    "rows": matrix.rows,
                    "cols": matrix.cols,
                    "nnz": matrix.nnz,
                    "kind": matrix.kind,
                    "url": matrix.url,
                }
            )
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/suitesparse"))
    parser.add_argument("--archive-dir", type=Path, default=Path("data/suitesparse_archives"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    for matrix in DEFAULT_MATRICES:
        archive = args.archive_dir / f"{matrix.stem}.tar.gz"
        download(matrix.url, archive)
        extract_matrix_market(archive, matrix, args.out_dir)
    write_manifest(args.out_dir / "suitesparse_manifest.csv", DEFAULT_MATRICES)


if __name__ == "__main__":
    main()

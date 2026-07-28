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


MATRIX_CATALOG = {
    "SNAP/ca-HepPh": SuiteSparseMatrix("SNAP", "ca-HepPh", 12008, 12008, 237010, "collaboration graph"),
    "SNAP/email-Enron": SuiteSparseMatrix("SNAP", "email-Enron", 36692, 36692, 367662, "email graph"),
    "SNAP/web-NotreDame": SuiteSparseMatrix("SNAP", "web-NotreDame", 325729, 325729, 1497134, "web graph"),
    "Bates/Chem97Zt": SuiteSparseMatrix("Bates", "Chem97Zt", 2541, 31022, 62044, "statistical/mathematical problem"),
    "LPnetlib/lp_fit2p": SuiteSparseMatrix("LPnetlib", "lp_fit2p", 3000, 13525, 50284, "linear programming problem"),
    "LPnetlib/lp_ken_18": SuiteSparseMatrix("LPnetlib", "lp_ken_18", 105127, 154699, 358171, "linear programming problem"),
    "LPnetlib/lp_stocfor3": SuiteSparseMatrix("LPnetlib", "lp_stocfor3", 16675, 23541, 72721, "linear programming problem"),
    "Meszaros/world": SuiteSparseMatrix("Meszaros", "world", 34506, 67147, 198883, "linear programming problem"),
    "Mittelmann/fome13": SuiteSparseMatrix("Mittelmann", "fome13", 48568, 97840, 285056, "linear programming problem"),
    "Meszaros/stat96v1": SuiteSparseMatrix("Meszaros", "stat96v1", 5995, 197472, 588798, "linear programming problem"),
    "NYPA/Maragal_6": SuiteSparseMatrix("NYPA", "Maragal_6", 21255, 10152, 537694, "least squares problem"),
    "Pereyra/landmark": SuiteSparseMatrix("Pereyra", "landmark", 71952, 2704, 1146848, "least squares problem"),
}


DEFAULT_MATRICES = [
    SuiteSparseMatrix("SNAP", "ca-HepPh", 12008, 12008, 237010, "collaboration graph"),
    SuiteSparseMatrix("SNAP", "email-Enron", 36692, 36692, 367662, "email graph"),
    SuiteSparseMatrix("SNAP", "web-NotreDame", 325729, 325729, 1497134, "web graph"),
]


def selected_matrices(names: list[str] | None) -> list[SuiteSparseMatrix]:
    if not names:
        return DEFAULT_MATRICES
    missing = [name for name in names if name not in MATRIX_CATALOG]
    if missing:
        available = ", ".join(sorted(MATRIX_CATALOG))
        raise SystemExit(f"unknown SuiteSparse matrix name(s): {', '.join(missing)}\navailable: {available}")
    return [MATRIX_CATALOG[name] for name in names]


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
    parser.add_argument("--include", nargs="*", default=None, help="SuiteSparse names such as SNAP/ca-HepPh")
    args = parser.parse_args()

    matrices = selected_matrices(args.include)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.archive_dir.mkdir(parents=True, exist_ok=True)
    for matrix in matrices:
        archive = args.archive_dir / f"{matrix.stem}.tar.gz"
        download(matrix.url, archive)
        extract_matrix_market(archive, matrix, args.out_dir)
    write_manifest(args.out_dir / "suitesparse_manifest.csv", matrices)


if __name__ == "__main__":
    main()

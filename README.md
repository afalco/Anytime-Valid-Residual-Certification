# Anytime-Valid Residual Certification

This repository contains the experimental code for the paper

> Anytime-Valid Residual Certification for Adaptive Randomized Low-Rank Approximation

The scripts reproduce the numerical studies used in the manuscript:

- repeated-inspection miscoverage for fixed-time certificates;
- adaptive randomized QB experiments on synthetic spectra;
- effective-rank sensitivity experiments;
- adaptive randomized QB experiments on structured real matrices;
- SuiteSparse sparse-matrix experiments with runtime and matrix-vector product
  cost accounting.

The code is intentionally lightweight and uses standard scientific Python
packages. Small real matrices used in the paper are generated locally from
reproducible scripts and standard example datasets. The SuiteSparse scripts
download selected Matrix Market files into an ignored local data directory.

## Repository Layout

```text
experiments/
  run_all_experiments.py
  miscoverage_study/
  synthetic_qb_study/
  real_matrix_study/
  src/
```

Each experiment directory writes its own `results/` and `figures/` folders when
run. These generated folders are ignored by Git.

## Installation

Using conda:

```bash
conda create -n anytime-qb python=3.12 numpy scipy scikit-learn scikit-image matplotlib
conda activate anytime-qb
```

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The optional SVG-to-PDF conversion step requires `rsvg-convert`.

## R1 Working Revision

The revision branch implements the certified/cap-reached output contract described
in `experiments/src/README.md`. Run its small tests with:

```bash
python -m unittest discover -s tests -v
```

Modified QB runners write under `revision_R1/` by default instead of overwriting
historical results. Their CLI accepts `--results-dir` for explicit output paths.
The synthetic and real-matrix runners also accept `--adaptive-c C` (C>4) as an
optional sketch-size variant. Power-law spending remains the default.

R1 changes basis expansion and skips probes at invalid denominators, changing
random streams and requiring fresh experimental tables. The previous tables are
not outputs of the revised implementation. Directional cost accounting and the
energy baseline are documented in `experiments/energy_baseline_study/README.md`;
the frozen production runs and regenerated resources are documented in
`experiments/revision_R1_study/README.md`. See `tests/README.md` for contract coverage.
The limits of the Gaussian implementation and the distinction between matrix
rank, tensor structure and prospective greedy extensions are recorded in
[`docs/R1_scope.md`](docs/R1_scope.md).

## Reproduce All Experiments

The command below is the historical aggregate runner. For the R1 production
pipeline use `experiments/revision_R1_study/README.md`, not this legacy command.

From the repository root:

```bash
python experiments/run_all_experiments.py
```

This regenerates the CSV files and figures used in the manuscript. Individual
experiments can also be run from their subdirectories; see the corresponding
`README.md` files.

SuiteSparse data are not stored in the repository. To download the matrices and
run the sparse study, see `experiments/real_matrix_study/README.md`.

## Citation

If you use this code, please cite the accompanying paper and this repository.
The repository citation is provided in `CITATION.cff`.

Repository URL:

```text
https://github.com/afalco/Anytime-Valid-Residual-Certification
```

# Anytime-Valid Residual Certification

This repository contains the experimental code for the paper

> Anytime-Valid Residual Certification for Adaptive Randomized Low-Rank Approximation

The scripts reproduce the numerical studies used in the manuscript:

- repeated-inspection miscoverage for fixed-time certificates;
- adaptive randomized QB experiments on synthetic spectra;
- effective-rank sensitivity experiments;
- adaptive randomized QB experiments on structured real matrices.

The code is intentionally lightweight and uses standard scientific Python
packages. Real matrices used in the paper are generated locally from
reproducible scripts and standard example datasets.

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

## Reproduce All Experiments

From the repository root:

```bash
python experiments/run_all_experiments.py
```

This regenerates the CSV files and figures used in the manuscript. Individual
experiments can also be run from their subdirectories; see the corresponding
`README.md` files.

## Citation

If you use this code, please cite the accompanying paper and this repository.
The repository citation is provided in `CITATION.cff`.

Repository URL:

```text
https://github.com/afalco/Anytime-Valid-Residual-Certification
```

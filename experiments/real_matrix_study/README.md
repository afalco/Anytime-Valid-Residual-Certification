# Real Matrix Study

This directory is for completing Section 7.4, `Real Matrices`.

## Purpose

The goal is to test the adaptive QB stopping rules on real or externally supplied matrices, rather than synthetic spectra.

The experiment compares:

- `naive`: fixed-time error level \(\alpha\) reused at every inspection;
- `anytime`: alpha-spending \(\alpha_j = 6\alpha/(\pi^2j^2)\).

For each matrix, the script records:

- dimensions;
- whether the input is sparse;
- selected rank;
- number of inspections;
- final true residual ratio;
- final certificate ratio;
- runtime;
- false stop indicator.

## Supported Matrix Formats

Place matrices in `data/`. If the directory does not exist yet, the script creates it and exits with a short message.

Supported formats:

- `.npy`: dense NumPy array.
- `.npz`: either a dense NumPy archive or a SciPy sparse matrix saved with `scipy.sparse.save_npz`.
- `.mtx`: Matrix Market format, loaded with `scipy.io.mmread`.

For sparse inputs, the script uses sparse matrix products where possible.

## Initial Benchmark Matrices

This directory includes a reproducible generator for a small initial matrix pack:

- `camera_like_192x192.npy`: dense image-like matrix with smooth regions, edges, and texture.
- `term_document_260x190.npz`: sparse term-document count matrix generated from latent topics.
- `elliptic_laplacian_289x289.npz`: sparse finite-difference elliptic operator.

Generate them from this directory with:

```bash
python generate_benchmark_matrices.py --out-dir data
```

The generator also writes `data/manifest.csv`.

## Application Matrices

The directory also includes a generator for small real matrices drawn from
standard Python example datasets:

- `digits_1797x64.npy`: centered handwritten-digit samples from scikit-learn.
- `skimage_camera_512x512.npy`: grayscale camera image.
- `skimage_coins_303x384.npy`: grayscale coins image.
- `skimage_astronaut_gray_384x384.npy`: resized grayscale astronaut image.

Generate them with:

```bash
python generate_application_matrices.py --out-dir data
```

This requires `scikit-learn` and `scikit-image`.

## Conda Environment

Recommended:

```bash
conda create -n anytime-qb python=3.12 numpy scipy pandas
conda activate anytime-qb
conda install scikit-learn scikit-image matplotlib
```

`pandas` is optional. The script writes CSVs using the standard library.

## Run

From the project root:

```bash
python experiments/real_matrix_study/run_real_matrices.py --data-dir experiments/real_matrix_study/data
```

Or from this directory:

```bash
python generate_benchmark_matrices.py --out-dir data
python generate_application_matrices.py --out-dir data
python run_real_matrices.py --data-dir data
python plot_real_matrix_ranks.py --results results/real_matrix_overhead.csv --out-dir figures
```

Useful options:

```bash
python experiments/real_matrix_study/run_real_matrices.py \
  --data-dir experiments/real_matrix_study/data \
  --trials 10 \
  --block-size 5 \
  --cert-size 80 \
  --alpha 0.05 \
  --rel-tol 0.1 \
  --rho-mode oracle
```

## Outputs

- `results/real_matrix_trials.csv`: per-matrix, per-method, per-trial results.
- `results/real_matrix_summary.csv`: aggregated results.
- `results/real_matrix_overhead.csv`: anytime-minus-naive overhead by matrix.
- `results/real_matrix_table.tex`: LaTeX table for the paper.
- `figures/real_matrix_rank_comparison.pdf`: rank comparison figure for the paper.

The generated table can be copied into the manuscript table directory, and the
generated PDF figure can be copied into the manuscript figure directory.

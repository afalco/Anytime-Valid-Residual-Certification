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
- total runtime;
- QB runtime, certification runtime, and final residual-evaluation runtime;
- matrix-vector products with \(A\), split into QB sampling, certification, and
  final residual evaluation;
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

## SuiteSparse Matrices

The paper-scale real-matrix study uses selected matrices from the SuiteSparse
Matrix Collection:

- `SNAP/ca-HepPh`: \(12008\times 12008\), 237010 nonzeros.
- `SNAP/email-Enron`: \(36692\times 36692\), 367662 nonzeros.
- `SNAP/web-NotreDame`: \(325729\times 325729\), 1497134 nonzeros.

Download them with:

```bash
python download_suitesparse_matrices.py --out-dir data/suitesparse
```

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

For the SuiteSparse study, use a non-oracle effective-rank mode to avoid
dense SVD computations:

```bash
python download_suitesparse_matrices.py --out-dir data/suitesparse
python run_real_matrices.py \
  --data-dir data/suitesparse \
  --trials 3 \
  --block-size 10 \
  --cert-size 200 \
  --cert-batch-size 20 \
  --max-rank 80 \
  --rho-mode conservative \
  --results-dir results/suitesparse
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

To run only selected SuiteSparse matrices, pass their file stems:

```bash
python run_real_matrices.py \
  --data-dir data/suitesparse \
  --include SNAP_ca-HepPh SNAP_email-Enron \
  --rho-mode conservative \
  --results-dir results/suitesparse_medium
```

## Outputs

- `results/real_matrix_trials.csv`: per-matrix, per-method, per-trial results.
- `results/real_matrix_summary.csv`: aggregated results.
- `results/real_matrix_overhead.csv`: anytime-minus-naive overhead by matrix.
- `results/real_matrix_table.tex`: LaTeX table for the paper.
- `results/real_matrix_cost_table.tex`: LaTeX cost table with runtime and matvec counts.
- `figures/real_matrix_rank_comparison.pdf`: rank comparison figure for the paper.

After running, send me the generated CSVs or just say it is done, and I can integrate the table into `paper.tex`.

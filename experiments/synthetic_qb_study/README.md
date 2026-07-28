# Synthetic QB Study

This experiment connects the anytime residual certification theory to an adaptive randomized QB procedure on synthetic matrices.

## Purpose

The experiment compares two stopping rules:

- `naive`: uses the same fixed-time error level \(\alpha\) at every inspection;
- `anytime`: uses alpha-spending \(\alpha_j = 6\alpha/(\pi^2j^2)\).

Both rules use fresh Gaussian certification sketches at every inspection. The script records the selected rank, the true residual, the certificate value, and whether the run stopped falsely.

## Important Scope

This is a controlled synthetic experiment. It is intended to test the mechanism and generate paper figures before importing the original paper's implementation. It uses a simple blocked randomized QB update:

\[
Y_j=(I-Q_jQ_j^\top)A\Omega_j,
\]

followed by orthogonalization and update of \(Q_j\).

## Files

- `run_synthetic_qb.py`: main experiment script.
- `run_effective_rank_sensitivity.py`: compares anytime stopping with an oracle effective-rank lower bound against the conservative \(\rho=1\) certificate.
- `results/synthetic_qb_summary.csv`: aggregated results.
- `results/synthetic_qb_trials.csv`: per-trial results.
- `figures/synthetic_qb_rank_overhead.svg`: generated SVG figure.
- `figures/synthetic_qb_rank_overhead.pdf`: generated PDF figure.

## Reproduce

From the project root:

```bash
python experiments/synthetic_qb_study/run_synthetic_qb.py
```

Optional arguments:

```bash
python experiments/synthetic_qb_study/run_synthetic_qb.py --trials 60 --n 120 --block-size 5 --cert-size 80 --alpha 0.05 --rel-tol 0.1
```

## Outputs

The main figure reports the mean selected rank by spectrum family and stopping rule. The CSV files contain the empirical false-stop rates and residual statistics.

## Effective-Rank Sensitivity

Run:

```bash
python experiments/synthetic_qb_study/run_effective_rank_sensitivity.py
```

This generates:

- `results/effective_rank_sensitivity_summary.csv`
- `results/effective_rank_sensitivity_overhead.csv`
- `results/effective_rank_sensitivity_table.tex`
- `figures/effective_rank_sensitivity.svg`

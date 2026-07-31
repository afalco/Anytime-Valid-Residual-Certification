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
- `run_adaptive_qb_coverage.py`: direct adaptive-QB coverage diagnostic at the data-dependent stopping rank.
- `run_effective_rank_sensitivity.py`: compares anytime stopping with an oracle effective-rank lower bound against the conservative \(\rho=1\) certificate.
- `results/synthetic_qb_summary.csv`: aggregated results.
- `results/synthetic_qb_trials.csv`: per-trial results.
- `figures/synthetic_qb_rank_overhead.svg`: generated SVG figure.
- `figures/synthetic_qb_rank_overhead.pdf`: generated PDF figure.

## Reproduce

From the project root:

```bash
/Users/antoniofalco/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 experiments/synthetic_qb_study/run_synthetic_qb.py
```

Optional arguments:

```bash
/Users/antoniofalco/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 experiments/synthetic_qb_study/run_synthetic_qb.py --trials 60 --n 120 --block-size 5 --cert-size 80 --alpha 0.05 --rel-tol 0.1
```

## Outputs

The main figure reports the mean selected rank by spectrum family and stopping rule. The CSV files contain the empirical false-stop rates and residual statistics.

## Adaptive-QB Coverage Diagnostic

Run:

```bash
/Users/antoniofalco/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 experiments/synthetic_qb_study/run_adaptive_qb_coverage.py
```

This stress-test diagnostic uses small certification sketches and \(\alpha=0.25\) to make repeated-inspection effects visible inside the adaptive QB loop. It reports two empirical failure rates at the selected rank:

- certificate miscoverage: the final certificate is below the exact squared residual;
- false stop: the algorithm accepts while \(\|R_\tau\|_{\mathrm{F}}>\varepsilon\).

The default table uses \(N=500\) trials, certification sketch size \(s=3\), tolerance \(\varepsilon=0.1\|A\|_{\mathrm{F}}\), block size \(1\), and maximum rank \(k_{\max}=80\). Failure rates are reported with Wilson 95% confidence intervals, so a zero observed failure rate is not presented as proof of zero probability. The table also reports the fraction of certified stops; runs that reach the rank cap without satisfying the certificate inequality are rank-cap terminations, not certified stops. The diagnostic uses oracle effective-rank information so that the table isolates sequential calibration from effective-rank estimation.

## Effective-Rank Sensitivity

Run:

```bash
/Users/antoniofalco/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 experiments/synthetic_qb_study/run_effective_rank_sensitivity.py
```

This generates:

- `results/effective_rank_sensitivity_summary.csv`
- `results/effective_rank_sensitivity_overhead.csv`
- `results/effective_rank_sensitivity_table.tex`
- `figures/effective_rank_sensitivity.svg`

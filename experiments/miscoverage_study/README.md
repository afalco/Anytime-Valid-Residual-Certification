# Miscoverage Study

This directory contains the first experiment for the revised paper.

## Purpose

The experiment illustrates the optional-stopping problem behind adaptive residual certification:

- a fixed-time certificate with nominal failure probability \(\alpha\) can fail with much larger probability when inspected repeatedly;
- an anytime-valid alpha-spending rule keeps the cumulative failure probability controlled.

This is a stylized experiment. It is not meant to model every dependency in randomized QB. Its role is to provide the opening empirical figure for the paper and to make the repeated-inspection effect visually clear.

## Files

- `run_miscoverage.py`: generates exact curves, Monte Carlo estimates, CSV output, and an SVG figure.
- `results/miscoverage_summary.csv`: generated numerical results.
- `figures/miscoverage_optional_stopping.svg`: generated figure.

## Reproduce

From the project root:

```bash
python experiments/miscoverage_study/run_miscoverage.py
```

Optional arguments:

```bash
python experiments/miscoverage_study/run_miscoverage.py --alpha 0.05 --max-checks 100 --trials 20000 --seed 0
```

## Model

For the naive repeated-inspection baseline, each inspection has false certification probability \(\alpha\). Under independence, the probability of at least one false certification by inspection \(T\) is

\[
1-(1-\alpha)^T.
\]

For anytime alpha-spending, the inspection-level budget is

\[
\alpha_j = \frac{6\alpha}{\pi^2j^2}.
\]

The cumulative union-bound guarantee is

\[
\sum_{j=1}^T\alpha_j \leq \alpha.
\]

The script reports both exact curves and Monte Carlo estimates.

The figure uses different line styles because several curves are expected to overlap closely:

- solid red: naive exact curve;
- dashed orange: naive Monte Carlo estimate;
- solid blue: anytime union bound;
- dotted green: anytime Monte Carlo estimate;
- dashed purple: anytime exact independent-failure curve.

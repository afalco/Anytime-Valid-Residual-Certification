# R1 follow-up experiments

This study supplements, and does not replace or mutate, the frozen 9910-trial
production study. `protocol.json` fixes the design before the follow-up runs.
The two-trial Schur pilot is only a feasibility/timing check, not production data.
The numerical design was retained unchanged for production.

## Explicit projection replication

2000 trials per rule, seed 20261108, n=80, s=3, alpha=.05, relative tolerance .7.
Uses actual adaptive Q updates and fresh Gaussian probes. Its first 500 trials
are checked against the historical production CSV when available. They are not
independent new observations; the expanded check has 2000, not 2500, trials.
The original 500-trial files and original protocol remain unchanged. Full-rank
zero is an ideal projector convention; numerical residual and orthogonality
errors are recorded separately. No repeated reseeding or extension until an
interval contains the analytical value is performed.

## Implicit Schur complement

The operator is `S = shift*I - C H^{-1} D`: an actual Schur complement with
`A11=H`, `A12=D`, `A21=C`, `A22=shift*I`. H is a sparse, nonuniform 1D elliptic
stiffness matrix plus a positive diagonal, and C/D are distinct sparse bidiagonal
couplings. Forward and transpose products each require one sparse triangular-solve
pair after LU factorization. The operator is not assembled. Details and seeds
are completely specified by the generator and protocol.

20 paired trials, n=1024, block size 5, cap 60, absolute epsilon=.01, s=80,
alpha=.05, power spending, rho=1. Absolute epsilon is fixed before any norm query.
Both methods use matching basis prefixes and final QB output. Energy scan pays
1024 forward RHS to compute the norm in blocks of 32, then maintains B.
Anytime power does not query the norm and forms B at output. Both pay k transpose
RHS for the coefficients. Setup and factorization time is included in solver
time and also reported separately. Runs are sequential with one BLAS thread;
method order is randomized per trial.

An exhaustive columnwise residual check is offline and separately charged 1024
forward RHS per method. Only n-by-batch temporaries are held, not the n-by-n
operator. Actual operator-side counters are checked against directional ledgers.
Shared-prefix checksums verify pairing.

This is a constructed compressible operator, not an application-data benchmark
or an information-theoretic lower bound. The comparison is against one explicit
column-scan energy implementation, not every specialized Schur norm algorithm.
Known norms, additional block structure or amortization across many QB runs can
favor deterministic stopping. Twenty trials measure cost, not validate a 5%
risk guarantee. Sparse float64 solves and energy arithmetic are not validated
outward-rounded computations.

## Run and rebuild

Use the repository's conda environment or install `requirements.txt`.
Run from the repository root (results are ignored by Git):

```bash
python experiments/revision_followup_study/run_followup.py --out-dir experiments/revision_followup_study/revision_R1/production_v1
python experiments/revision_followup_study/build_followup.py --results experiments/revision_followup_study/revision_R1/production_v1 --manuscript-dir /absolute/path/to/research-paper-improvement/manuscript/revision_R1
python experiments/revision_R1_study/build_artifacts.py --results experiments/revision_R1_study/revision_R1/production_v1_full --followup-results experiments/revision_followup_study/revision_R1/production_v1 --manuscript-dir /absolute/path/to/research-paper-improvement/manuscript/revision_R1
```

The first command refuses to overwrite results. `--group explicit` or
`--group implicit` runs a group separately. Use a new output directory for
`--pilot-trials 2 --group implicit`. Do not use pilot output to build tables.
`build_followup.py` validates manifests, source hashes, row counts and ledgers;
`build_artifacts.py --followup-results` uses the expanded explicit check in the
manuscript while preserving the original result files. Omitting that option
intentionally rebuilds the historical 500-trial check instead.

Scripts: `run_followup.py` implements both studies; `build_followup.py` validates
and summarizes them. Manifests record source hashes, seeds, versions and thread
settings. `summary.json` records phase timings, rates/intervals and table hashes.
Tests are in `tests/test_R1_followup.py`.

# Energy-identity baseline and R1 access costs

This study compares energy stopping with Gaussian residual certification under a
stored-matrix model. It is not evidence that randomized certification is needed
when the matrix norm and exact coefficients are already available.

Run from the repository root, in the project's Python environment:

```bash
python experiments/energy_baseline_study/run_energy_baseline.py
```

The defaults are a small pilot: n=80, 3 trials for each of three synthetic spectra,
block size 5, rank cap 60, certification size 80, alpha=0.05, relative tolerance
0.1, seed 20261107. To use stored matrices instead:

```bash
python experiments/energy_baseline_study/run_energy_baseline.py \
  --matrix-paths /absolute/path/matrix.npy /absolute/path/matrix.npz \
  --trials 3 --max-rank 30
```

NumPy and SciPy are required. Large direct diagnostics densify at most 64 columns
at a time and still perform O(m*n) dense residual arithmetic; use this pilot on
manageable matrices before planning a large sparse run.

## Comparisons and coupling

- `energy_incremental`: update B=Q^T A as independent basis columns are appended;
  stop using ||A||_F^2-||B||_F^2. This is a floating-point energy baseline, not an
  outward-rounded rigorous numerical bound.
- `naive_final`: fixed-time Gaussian risk reused; form B only at output.
- `anytime_final`: power-law risk spending; form B only at output.
- `anytime_incremental`: the same anytime rule while maintaining B throughout.

Every method resets a separate basis RNG to the same seed per matrix/trial, so
it follows the same Q prefix, independently of certification draws. Each run
executes only until its own stop; later basis work is not charged to it.
The two anytime coefficient modes also share their certification seed to isolate
coefficient scheduling from sketch noise. Execution order is randomized by trial.
RNG seeds, order, configuration, library versions, thread environment, code hashes,
and supplied data hashes are saved. Array loading and synthetic data generation
are outside solver timing; the input norm is recomputed and charged per run.

## Cost contract

`solver_A` and `solver_AT` count right-hand sides, not block calls. The JSON ledger
in each trial separates qb, b, certification, pilot, oracle, norm, and diagnostic.
Phase timers do not overlap. `solver_runtime_sec` includes preparation after input
validation, norm, basis, coefficients, and stopping work; `runtime_sec` additionally
includes the offline residual diagnostic. Input validation and data loading are
not included. Per-phase times omit loop/dispatch overhead, so they need not sum
exactly to the solver wall-clock time.

For block width w, actual retained rank increment r, sketch s and pilot p:

| Phase | A | A transpose |
|---|---:|---:|
| Basis sampling | w | 0 |
| Coefficient update | 0 | r |
| Valid Gaussian certification | s | 0 |
| Invalid denominator / known zero | 0 | 0 |
| Effective-rank pilot | p | at most p |
| Cached energy diagnostic | 0 | 0 |

Output B costs k transpose products whether constructed incrementally or once at
the final retained rank k. Incremental construction assumes old Q columns are
unchanged; cache updates reject a changed prefix. In-place mutation of a retained
basis is unsupported. The current expansion preserves earlier columns.

`norm_entries_read` is the stored-entry count for the norm pass, not a conversion
to operator products and not a count of every validation or cache access.
`oracle_entries_materialized` and its timer explicitly expose full-residual/oracle
work; oracle results must not be called a matrix-free algorithm. The baseline
driver uses rho=1 and no oracle or heuristic pilot.

The offline `direct` diagnostic evaluates stored blocks of A-QB using the already
formed B. It is independent of the energy subtraction, counts zero operator
applications, records m*n materialized entries over all blocks, and is not free:
its elapsed time is separate. An `energy` diagnostic reuses B without another A^T Q
but is not an independent accuracy check of the energy baseline.

## Outputs and interpretation

`revision_R1/pilot/` (or `--out-dir`) contains config.json, trials.csv, summary.csv.
Counts and timing components are retained for cap exits as well as certified
stops. A cap exit is never silently removed from cost averages. Zero observed
false stops in three trials does not validate a risk level. These pilots do not
replace production R1 tables and are not automatically inserted into the paper.

The older synthetic rank/coverage scripts materialize residuals for diagnostics;
they are not used as wall-clock or matrix-free cost benchmarks. This driver uses
the instrumented real-matrix engine for both synthetic and stored real inputs.

# R1 contract tests

Run from the repository root:

```bash
python -m unittest discover -s tests -v
```

Requires NumPy and SciPy. Tests cover denominator boundaries and zero probe use
at invalid denominators, known-zero inputs, bounded inspections after rank
deficiency, orthonormality, partial final blocks, last-inspection acceptance,
false-stop flags, adaptive sketch-size positivity, and reporting denominators.

Mocked inaccurate certificates test the output contract; they are not claims
that the Gaussian bound fails in that deterministic way. Numerical residual
evaluation is an offline diagnostic, not an exact-zero detection rule. These
tests are not production Monte Carlo experiments or performance benchmarks.

`test_cost_accounting.py` additionally checks right-hand-side counts for both
operator directions, nonoverlapping timers, incremental/final coefficients,
rotation rejection, rectangular/sparse matrices, cached diagnostics without
double counting, pilot directions, matched Q prefixes and offline energy checks.

`test_R1_discussion_examples.py` checks Rademacher moments by exhaustive sign
enumeration, the generated Laplacian's Kronecker sum and positive spectrum,
and the SPD residual-to-solution-error inequality used in the discussion.
These are illustrative algebra checks, not new distributional guarantees.

`test_R1_followup.py` compares Schur forward and transpose products with a small
explicit reference, checks adjoint consistency and actual directional counters,
and verifies paid norm acquisition versus absolute-tolerance stopping without
a norm query. It also checks matched basis prefixes and offline-query separation.

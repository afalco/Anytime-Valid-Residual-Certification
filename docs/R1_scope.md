# Scope of the R1 certification framework

The implemented and benchmarked residual certificates use Gaussian probes.
The sequential error-spending argument is distribution-independent, but a new
sketch must supply a conditional fixed-time bound for the residual selected
before validation. Unbiasedness by itself is not a certificate. Do not replace
Gaussian probes with signs, coordinates or structured transforms while retaining
the Gaussian denominator or chi-square calibration.

The manuscript discusses Rademacher variance, coordinate concentration and
within-block dependence as examples of the analysis that alternative sketches
require. None is an additional benchmarked implementation.

The generated five-point Dirichlet Laplacian is the Kronecker sum
`I_q (x) T_q + T_q (x) I_q`, with `T_q = tridiag(-1, 2, -1)` and `q = 17`.
It is full rank as a 289-by-289 matrix. Its two-term separated operator
representation does not imply rank-two matrix structure or small solution tensor
ranks. The QB benchmark measures matrix approximation, not a tensor solver.

Greedy rank-one/PGD certification is prospective. It needs a specified equation
residual norm, a stability bound if solution error is the target, conditional
validation independent of direction selection and inner solves, and explicit
control of compression and contraction errors. Rank-one tensor probes are not
vectors of independent Gaussian entries, and greedy updates need not satisfy the
orthogonal-projection energy identity.

`tests/test_R1_discussion_examples.py` checks the finite-dimensional identities
used to explain these distinctions. It is not a new Monte Carlo study or a proof
of a general non-Gaussian or PGD certificate. Production configurations and raw
data from Block 3 are unchanged by these documentation additions.

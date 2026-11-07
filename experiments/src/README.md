# Shared certificate contract (R1)

`certification.py` supplies the shared Gaussian guard, predictable sketch-size
rule, parameter validation, and rank-revealing basis expansion.

- A nonpositive denominator gives an infinite bound with zero certification probes.
- `known_zero=True` requires an independently justified structural zero; runners
  only recognize the exactly zero input matrix this way.
- A small sketch or nominally full basis is not an exact-zero test.
- SVD-based expansion drops unresolved dependent directions instead of adding
  columns through QR of a rank-deficient sample; two projection passes precede it.
- Both rank and inspections are capped, with J=ceil(rank_cap/block_size).
  Rank-deficient blocks can exhaust the inspection cap below the rank cap.
- `certified` and `cap_reached` are exclusive. `tolerance_met` is an offline
  numerical diagnostic. `false_stop` requires certified acceptance.
- `s_requested` records pre-sketch selected sizes; `s_used` records actual probes.
- Optional adaptive c>4 ensures positivity, not certified termination.

R1 changes random-number consumption and basis expansion relative to the original
scripts. Old tables must not be relabeled as R1 results. `cost_accounting.py` now
provides directional operator counts, phase timers, coefficient caching, and
energy/direct diagnostics for the real-matrix engine. The paired stored-matrix
baseline uses this engine for both synthetic and real inputs; its README is in
`experiments/energy_baseline_study/`. Production results remain pending.
Test instructions are in tests/README.md.

"""Operator applications and disjoint phase timers for R1 stored-matrix studies."""

from contextlib import contextmanager
import math
from time import perf_counter

import numpy as np


PHASES = ('qb', 'b', 'certification', 'pilot', 'oracle', 'norm', 'diagnostic')


class CostLedger:
    def __init__(self):
        self.counts = {phase: {'A': 0, 'AT': 0} for phase in PHASES}
        self.seconds = {phase: 0.0 for phase in PHASES}
        self._active = None

    @contextmanager
    def measure(self, phase):
        if phase not in PHASES or self._active is not None:
            raise ValueError('phase timers must be valid and nonoverlapping')
        self._active = phase
        start = perf_counter()
        try:
            yield
        finally:
            self.seconds[phase] += perf_counter() - start
            self._active = None

    def apply(self, a, x, phase, *, transpose=False):
        if phase not in PHASES or x.ndim not in (1, 2):
            raise ValueError('invalid phase or right-hand side shape')
        operator = a.T if transpose else a
        if x.shape[0] != operator.shape[1]:
            raise ValueError('operator and right-hand side dimensions do not match')
        value = np.asarray(operator @ x)
        self.counts[phase]['AT' if transpose else 'A'] += 1 if x.ndim == 1 else x.shape[1]
        return value

    def total(self, direction, *, diagnostic=False):
        return sum(count[direction] for phase, count in self.counts.items()
                   if (phase == 'diagnostic') == diagnostic)


class CoefficientCache:
    """Incremental A^T Q, valid only when previously retained columns are unchanged."""

    def __init__(self, columns):
        self.bt = np.empty((columns, 0))
        self.q = None

    def update(self, a, q, ledger):
        old = self.bt.shape[1]
        if q.shape[1] < old or (self.q is not None and not np.array_equal(q[:, :old], self.q)):
            raise ValueError('basis changed: rebuild coefficients after a rotation or reorthogonalization')
        if q.shape[1] > old:
            added = ledger.apply(a, q[:, old:], 'b', transpose=True)
            self.bt = np.column_stack((self.bt, added))
        self.q = q
        return self.bt


def energy_residual(norm_a, bt):
    """Floating-point energy identity, not an outward-rounded certificate."""
    captured = float(np.sum(bt * bt))
    raw = norm_a * norm_a - captured
    if not math.isfinite(raw):
        raise FloatingPointError('nonfinite energy difference')
    return math.sqrt(max(raw, 0.0)), raw < 0


def direct_stored_residual(a, q, bt, batch_size=64):
    """Independent stored-entry diagnostic; no operator applications are hidden."""
    total = 0.0
    if hasattr(a, 'tocsc'):
        a = a.tocsc(copy=False)
    for first in range(0, a.shape[1], batch_size):
        last = min(first + batch_size, a.shape[1])
        block = a[:, first:last]
        if hasattr(block, 'toarray'):
            block = block.toarray()
        residual = np.asarray(block) - q @ bt[first:last, :].T
        total += float(np.sum(residual * residual))
    if not math.isfinite(total):
        raise FloatingPointError('nonfinite direct residual energy')
    return math.sqrt(total)

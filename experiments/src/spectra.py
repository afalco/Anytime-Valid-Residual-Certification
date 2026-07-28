"""Synthetic spectra for low-rank approximation experiments."""

from __future__ import annotations

import numpy as np


def polynomial_decay(n: int, power: float = 1.0) -> np.ndarray:
    i = np.arange(1, n + 1, dtype=float)
    return i ** (-power)


def exponential_decay(n: int, rate: float = 0.95) -> np.ndarray:
    i = np.arange(n, dtype=float)
    return rate ** i


def flat_tail(n: int, rank: int, head: float = 1.0, tail: float = 1e-2) -> np.ndarray:
    sigma = np.full(n, tail, dtype=float)
    sigma[:rank] = head
    return sigma


def effective_rank_from_spectrum(sigma: np.ndarray) -> float:
    energy = np.sum(sigma**2)
    fourth = np.sum(sigma**4)
    return float((energy**2) / fourth)

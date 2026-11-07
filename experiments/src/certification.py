"""Shared R1 certificate guards and rank-revealing basis expansion."""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import Callable

import numpy as np


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f'{name} must be a positive integer')
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if result < 1:
        raise ValueError(f'{name} must be a positive integer')
    return result


def inflation_factor(alpha_level: float, cert_size: int, rho: float) -> float:
    positive_int(cert_size, 'cert_size')
    if not math.isfinite(alpha_level) or not 0 < alpha_level < 1:
        raise ValueError('alpha_level must be in (0,1)')
    if not math.isfinite(rho) or rho < 1:
        raise ValueError('rho must be finite and >=1')
    return 2 * math.sqrt(-math.log(alpha_level) / cert_size / rho)


def sketch_size(alpha_level: float, minimum: int, rho: float, adaptive_c: float | None = None) -> int:
    inflation_factor(alpha_level, minimum, rho)
    if adaptive_c is None:
        return minimum
    if not math.isfinite(adaptive_c) or adaptive_c <= 4:
        raise ValueError('adaptive_c must be finite and >4')
    return max(minimum, math.ceil(adaptive_c * (-math.log(alpha_level)) / rho))


@dataclass(frozen=True)
class CertificateResult:
    upper: float
    denominator_valid: bool
    probes_used: int


def gaussian_certificate(
    rng: np.random.Generator,
    residual_product: Callable[[np.ndarray], np.ndarray],
    columns: int,
    alpha_level: float,
    cert_size: int,
    rho: float,
    batch_size: int | None = None,
    *,
    known_zero: bool = False,
) -> CertificateResult:
    inflation = inflation_factor(alpha_level, cert_size, rho)
    batch_size = cert_size if batch_size is None else positive_int(batch_size, 'batch_size')
    if known_zero:
        return CertificateResult(0.0, True, 0)
    if inflation >= 1:
        return CertificateResult(math.inf, False, 0)
    total = 0.0
    for start in range(0, cert_size, batch_size):
        width = min(batch_size, cert_size - start)
        omega = rng.standard_normal((columns, width))
        values = residual_product(omega)
        total += float(np.sum(values * values))
    if not math.isfinite(total):
        raise FloatingPointError('nonfinite residual sketch energy')
    return CertificateResult(total / cert_size / (1 - inflation), True, cert_size)


def extend_basis(q: np.ndarray, samples: np.ndarray, scale: float) -> np.ndarray:
    """Retain resolved independent directions; never invent columns after breakdown."""
    y = samples
    for _ in range(2):
        if q.size:
            y = y - q @ (q.T @ y)
    u, singular, _ = np.linalg.svd(y, full_matrices=False)
    cutoff = np.finfo(float).eps * max(q.shape[0], q.shape[1] + y.shape[1]) * scale
    new = u[:, singular > cutoff]
    return np.column_stack([q, new])


def validate_run(method: str, rho_mode: str, modes: tuple[str, ...], block_size: int,
                 max_rank: int, rel_tol: float, alpha: float, cert_size: int) -> None:
    if method not in ('naive', 'anytime') or rho_mode not in modes:
        raise ValueError('unknown stopping method or effective-rank mode')
    positive_int(block_size, 'block_size')
    positive_int(max_rank, 'max_rank')
    if not math.isfinite(rel_tol) or rel_tol < 0:
        raise ValueError('rel_tol must be finite and nonnegative')
    inflation_factor(alpha, cert_size, 1.0)


def tolerance_ratio(value: float, tolerance: float) -> float:
    if tolerance > 0:
        return value / tolerance
    return 0.0 if value == 0 else math.inf


def inspection_risk(alpha: float, inspection: int, horizon: int, schedule: str) -> float:
    positive_int(inspection, 'inspection')
    positive_int(horizon, 'horizon')
    if not 0 < alpha < 1 or inspection > horizon:
        raise ValueError('invalid risk budget or inspection outside declared horizon')
    if schedule == 'power':
        return 6 * alpha / (math.pi**2 * inspection**2)
    if schedule == 'uniform':
        return alpha / horizon
    raise ValueError('unknown risk schedule')

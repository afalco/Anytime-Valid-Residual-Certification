"""Stylized optional-stopping simulation for residual certificates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    alpha: float = 0.05
    max_checks: int = 100
    trials: int = 10000
    seed: int = 0


def run_simulation(config: SimulationConfig) -> dict[str, float]:
    rng = np.random.default_rng(config.seed)

    # Stylized model: each fixed-time certificate has false positive probability alpha.
    # Repeated inspection stops at the first false certificate.
    fixed_time_failures = rng.random((config.trials, config.max_checks)) < config.alpha
    stopped_wrongly = fixed_time_failures.any(axis=1)

    alpha_spend = 6 * config.alpha / (np.pi**2 * (np.arange(1, config.max_checks + 1) ** 2))
    anytime_failures = rng.random((config.trials, config.max_checks)) < alpha_spend
    anytime_wrongly = anytime_failures.any(axis=1)

    return {
        "naive_failure_rate": float(stopped_wrongly.mean()),
        "anytime_failure_rate": float(anytime_wrongly.mean()),
        "nominal_alpha": config.alpha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-checks", type=int, default=100)
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run_simulation(
        SimulationConfig(
            alpha=args.alpha,
            max_checks=args.max_checks,
            trials=args.trials,
            seed=args.seed,
        )
    )
    for key, value in result.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()

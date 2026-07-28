"""Miscoverage experiment for repeated residual certificate inspection.

The script intentionally uses only the Python standard library so that it is
easy to run on a clean machine. It writes:

- results/miscoverage_summary.csv
- figures/miscoverage_optional_stopping.svg
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path


def alpha_spending(alpha: float, max_checks: int) -> list[float]:
    return [6.0 * alpha / (math.pi**2 * j**2) for j in range(1, max_checks + 1)]


def simulate_any_failure(probabilities: list[float], trials: int, rng: random.Random) -> list[float]:
    counts = [0] * len(probabilities)
    for _ in range(trials):
        failed = False
        for idx, p in enumerate(probabilities):
            if not failed and rng.random() < p:
                failed = True
            if failed:
                counts[idx] += 1
    return [count / trials for count in counts]


def cumulative_any_failure_independent(probabilities: list[float]) -> list[float]:
    survival = 1.0
    values = []
    for p in probabilities:
        survival *= 1.0 - p
        values.append(1.0 - survival)
    return values


def cumulative_spending_bound(probabilities: list[float]) -> list[float]:
    total = 0.0
    values = []
    for p in probabilities:
        total += p
        values.append(total)
    return values


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checks",
        "naive_exact",
        "naive_monte_carlo",
        "anytime_exact_independent",
        "anytime_union_bound",
        "anytime_monte_carlo",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_polyline(
    points: list[tuple[float, float]],
    color: str,
    width: float = 2.5,
    dasharray: str | None = None,
) -> str:
    data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash = f' stroke-dasharray="{dasharray}"' if dasharray else ""
    return (
        f'<polyline points="{data}" fill="none" stroke="{color}" '
        f'stroke-width="{width}"{dash} stroke-linejoin="round" stroke-linecap="round" />'
    )


def svg_markers(
    points: list[tuple[float, float]],
    color: str,
    shape: str,
    every: int,
    radius: float = 4.0,
) -> list[str]:
    elements = []
    for idx, (x, y) in enumerate(points):
        if idx % every != 0 and idx != len(points) - 1:
            continue
        if shape == "circle":
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
                f'fill="white" stroke="{color}" stroke-width="2" />'
            )
        elif shape == "square":
            side = 2.0 * radius
            elements.append(
                f'<rect x="{x-radius:.2f}" y="{y-radius:.2f}" width="{side:.2f}" height="{side:.2f}" '
                f'fill="white" stroke="{color}" stroke-width="2" />'
            )
        elif shape == "diamond":
            elements.append(
                f'<path d="M {x:.2f},{y-radius:.2f} L {x+radius:.2f},{y:.2f} '
                f'L {x:.2f},{y+radius:.2f} L {x-radius:.2f},{y:.2f} Z" '
                f'fill="white" stroke="{color}" stroke-width="2" />'
            )
    return elements


def write_svg(path: Path, rows: list[dict[str, float]], alpha: float, max_checks: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 920, 560
    left, right, top, bottom = 82, 30, 45, 78
    plot_w = width - left - right
    plot_h = height - top - bottom

    def xmap(checks: float) -> float:
        return left + (checks - 1.0) * plot_w / max(1.0, max_checks - 1.0)

    def ymap(value: float) -> float:
        return top + (1.0 - value) * plot_h

    series = [
        # Draw Monte Carlo first, then exact/bound curves on top. Markers make
        # overlapped Monte Carlo curves visible in the generated PDF.
        ("naive_monte_carlo", "#f97316", "Naive fixed-time, Monte Carlo", "7 5", 2.2, "circle"),
        ("naive_exact", "#b42318", "Naive fixed-time, exact", None, 3.0, None),
        ("anytime_monte_carlo", "#079455", "Anytime alpha-spending, Monte Carlo", "2 5", 2.2, "square"),
        ("anytime_exact_independent", "#6941c6", "Anytime alpha-spending, exact", "8 4", 2.2, "diamond"),
        ("anytime_union_bound", "#155eef", "Anytime union bound", None, 3.0, None),
    ]

    elements: list[str] = []
    elements.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    elements.append('<rect width="100%" height="100%" fill="white"/>')
    elements.append(f'<text x="{left}" y="24" font-family="Helvetica, Arial, sans-serif" font-size="18" font-weight="700">Repeated inspection inflates fixed-time certificate failure</text>')
    elements.append(f'<text x="{left}" y="44" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#475467">Nominal alpha = {alpha:.3f}; alpha-spending uses 6 alpha / (pi^2 j^2); overlapping curves indicate agreement</text>')

    # Grid and labels.
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = ymap(tick)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#eaecf0" stroke-width="1"/>')
        elements.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#475467">{tick:.2f}</text>')
    for tick in [1, max_checks // 4, max_checks // 2, 3 * max_checks // 4, max_checks]:
        if tick < 1:
            continue
        x = xmap(tick)
        elements.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#f2f4f7" stroke-width="1"/>')
        elements.append(f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#475467">{tick}</text>')

    elements.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#344054" stroke-width="1.2"/>')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#344054" stroke-width="1.2"/>')
    elements.append(f'<text x="{left + plot_w/2:.2f}" y="{height-24}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#344054">Number of inspections</text>')
    elements.append(f'<text x="20" y="{top + plot_h/2:.2f}" transform="rotate(-90 20 {top + plot_h/2:.2f})" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#344054">Probability of at least one false certification</text>')

    point_cache: dict[str, list[tuple[float, float]]] = {}
    for key, color, _label, dasharray, line_width, _marker in series:
        pts = [(xmap(row["checks"]), ymap(row[key])) for row in rows]
        point_cache[key] = pts
        elements.append(svg_polyline(pts, color, line_width, dasharray))

    for key, color, _label, _dasharray, _line_width, marker in series:
        if marker:
            elements.extend(svg_markers(point_cache[key], color, marker, every=max(1, max_checks // 10)))

    legend_x, legend_y = left + 420, top + 58
    for idx, (_key, color, label, dasharray, _line_width, marker) in enumerate(series):
        y = legend_y + idx * 22
        dash = f' stroke-dasharray="{dasharray}"' if dasharray else ""
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+28}" y2="{y}" stroke="{color}" stroke-width="3"{dash} stroke-linecap="round"/>')
        if marker:
            elements.extend(svg_markers([(legend_x + 14, y)], color, marker, every=1, radius=3.6))
        elements.append(f'<text x="{legend_x+38}" y="{y+4}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#344054">{label}</text>')

    elements.append("</svg>")
    path.write_text("\n".join(elements))


def build_rows(alpha: float, max_checks: int, trials: int, seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    naive_probs = [alpha] * max_checks
    anytime_probs = alpha_spending(alpha, max_checks)

    naive_exact = cumulative_any_failure_independent(naive_probs)
    anytime_exact = cumulative_any_failure_independent(anytime_probs)
    anytime_bound = cumulative_spending_bound(anytime_probs)
    naive_mc = simulate_any_failure(naive_probs, trials, rng)
    anytime_mc = simulate_any_failure(anytime_probs, trials, rng)

    rows = []
    for idx in range(max_checks):
        rows.append(
            {
                "checks": idx + 1,
                "naive_exact": naive_exact[idx],
                "naive_monte_carlo": naive_mc[idx],
                "anytime_exact_independent": anytime_exact[idx],
                "anytime_union_bound": anytime_bound[idx],
                "anytime_monte_carlo": anytime_mc[idx],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-checks", type=int, default=100)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    rows = build_rows(args.alpha, args.max_checks, args.trials, args.seed)
    write_csv(base / "results" / "miscoverage_summary.csv", rows)
    write_svg(base / "figures" / "miscoverage_optional_stopping.svg", rows, args.alpha, args.max_checks)

    final = rows[-1]
    print(f"wrote: {base / 'results' / 'miscoverage_summary.csv'}")
    print(f"wrote: {base / 'figures' / 'miscoverage_optional_stopping.svg'}")
    print(f"naive final exact: {final['naive_exact']:.6f}")
    print(f"anytime final union bound: {final['anytime_union_bound']:.6f}")
    print(f"anytime final Monte Carlo: {final['anytime_monte_carlo']:.6f}")


if __name__ == "__main__":
    main()

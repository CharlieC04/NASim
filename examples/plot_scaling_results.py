"""
Plots examples/qldpc_scaling_results.csv (produced by
qldpc_scaling_experiment.py): qubit count on the x-axis, spacetime-volume
speedup (baseline_volume / qldpc_volume) on the y-axis, one line per
circuit family, plus a y=1 reference line (break-even -- above it qLDPC
wins, below it the surface-code-only baseline wins).

Run from the repo root, in the MAIN venv:

    python examples/plot_scaling_results.py

Needs matplotlib; if not already installed: pip install matplotlib
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_CSV = Path("examples/qldpc_scaling_results.csv")
OUT_PNG = Path("examples/qldpc_scaling_results.png")


def main() -> None:
    if not RESULTS_CSV.exists():
        raise SystemExit(f"{RESULTS_CSV} not found -- run qldpc_scaling_experiment.py first")

    by_family: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            by_family[row["family"]].append((int(row["n"]), float(row["speedup"])))

    fig, ax = plt.subplots(figsize=(8, 6))
    for family, points in sorted(by_family.items()):
        points.sort(key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", label=family)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="break-even")
    ax.set_xlabel("circuit size (qubits)")
    ax.set_ylabel("speedup (baseline volume / qLDPC volume)")
    ax.set_title("qLDPC memory hierarchy vs. surface-code-only baseline")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

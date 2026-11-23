"""
Plots examples/qldpc_fidelity_results.csv (produced by
qldpc_fidelity_calibration.py): physical error rate on the x-axis,
per-round logical error rate on the y-axis, log-log (qSIEVE's own Fig 8
style), one line per calibrated code, plus a y=x reference line (below it
the code is suppressing errors; above it, physical noise dominates and
more rounds/qubits would make things worse, not better).

Error bars are simple binomial (sqrt(n_errors))/shots -- a rough estimate
of shot-noise uncertainty, not a rigorous confidence interval, but enough
to see whether two points differ meaningfully or whether more shots are
needed before drawing conclusions.

A point with 0 observed errors can't be placed on a log axis (and "0
errors in N shots" isn't the same claim as "the rate is 0" anyway) -- it's
shown instead as a downward arrow at the rule-of-three upper bound
(~3/shots, ~95% CI for zero events), meaning "the true rate is very
likely below this arrow's tip, but we don't have enough shots to say more."

Also overlays qSIEVE's own Fig 8(a) curve for [[144,12,12]] as a reference
-- visually read off the published figure at p=6e-3..9e-3. (An earlier
version of this script used a pixel-analysis estimate instead; its 8e-3
value (~5.3e-2) was close to this one (5e-2), but its 6e-3 value was an
extrapolation through a region where the [144,12,12] and [1452,12,11]
curves visually overlap and is superseded by this direct reading.) Treat
these as approximate, not digitized ground truth.

Run from the repo root, in the MAIN venv:

    python examples/plot_fidelity_results.py

Needs matplotlib; if not already installed: pip install matplotlib
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_CSV = Path("examples/qldpc_fidelity_results.csv")
OUT_PNG = Path("examples/qldpc_fidelity_results.png")

# qSIEVE Fig 8(a), [[144,12,12]] (Weight 6) curve, read directly from the
# published figure -- see module docstring for provenance/caveats.
QSIEVE_144_12_12_REFERENCE = [
    (6e-3, 6e-3),
    (7e-3, 1.9e-2),
    (8e-3, 4.7e-2),
    (9e-3, 9.5e-2),
]


def main() -> None:
    if not RESULTS_CSV.exists():
        raise SystemExit(f"{RESULTS_CSV} not found -- run qldpc_fidelity_calibration.py first")

    # Keyed by (code, p_phys, rounds): the CSV accumulates incrementally
    # across runs, so re-running the calibration script at the same point
    # (e.g. after a fix) adds a new row rather than replacing the old one.
    # Keep only the latest row per (code, p_phys, rounds) so a stale
    # pre-fix result doesn't get plotted alongside its corrected rerun.
    latest: dict[tuple[str, float, int], tuple[int, int]] = {}
    with open(RESULTS_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["code"], float(row["p_phys"]), int(row["rounds"]))
            latest[key] = (int(row["n_errors"]), int(row["shots"]))

    by_code: dict[str, list[tuple[float, int, int, int]]] = defaultdict(list)
    for (code, p_phys, rounds), (n_errors, shots) in latest.items():
        by_code[code].append((p_phys, rounds, n_errors, shots))

    all_xs_for_range: list[float] = []
    all_ys_for_range: list[float] = []

    fig, ax = plt.subplots(figsize=(7, 6))
    for code, points in sorted(by_code.items()):
        points.sort(key=lambda pt: pt[0])
        xs, ys, yerr = [], [], []
        ub_xs, ub_ys = [], []  # zero-error points: plotted as upper bounds, not a rate
        for p_phys, rounds, n_err, shots in points:
            if n_err == 0:
                # Can't show 0 on a log axis, and "0 errors in N shots" isn't
                # the same claim as "the rate is 0" -- use the standard
                # rule-of-three upper bound (~95% CI) instead: the true rate
                # is very unlikely to exceed 3/shots, per-circuit, converted
                # to per-round the same way as the real points below.
                ub_rate = 3.0 / shots
                ub_per_round = 1 - (1 - ub_rate) ** (1 / rounds)
                ub_xs.append(p_phys)
                ub_ys.append(ub_per_round)
                all_xs_for_range.append(p_phys)
                all_ys_for_range.append(ub_per_round)
                continue

            rate = n_err / shots
            rate_err = math.sqrt(max(rate * (1 - rate), 1e-12) / shots)

            # propagate shot-noise through p_L = 1 - (1 - rate)^(1/rounds)
            # via d(p_L)/d(rate) = (1/rounds) * (1-rate)^(1/rounds - 1)
            per_round = 1 - (1 - rate) ** (1 / rounds)
            slope = (1 - rate) ** (1 / rounds - 1) / rounds

            xs.append(p_phys)
            ys.append(per_round)
            yerr.append(rate_err * slope)
            all_xs_for_range.append(p_phys)
            all_ys_for_range.append(per_round)

        if xs:
            line = ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=code)
        if ub_xs:
            color = line[0].get_color() if xs else None
            ax.plot(ub_xs, ub_ys, marker="v", linestyle="none", color=color,
                    label=None if xs else code,
                    markerfacecolor="none")
            for x, y in zip(ub_xs, ub_ys):
                ax.annotate("", xy=(x, y * 0.3), xytext=(x, y),
                            arrowprops=dict(arrowstyle="->", color=color or "C0", alpha=0.6))

    if QSIEVE_144_12_12_REFERENCE:
        ref_xs = [pt[0] for pt in QSIEVE_144_12_12_REFERENCE]
        ref_ys = [pt[1] for pt in QSIEVE_144_12_12_REFERENCE]
        ax.plot(ref_xs, ref_ys, marker="x", linestyle=":", color="black",
                label="qSIEVE Fig 8(a) [[144,12,12]] (approx., read from figure)")
        all_xs_for_range.extend(ref_xs)
        all_ys_for_range.extend(ref_ys)

    ax.set_xscale("log")
    ax.set_yscale("log")

    if all_xs_for_range:
        xlo, xhi = min(all_xs_for_range), max(all_xs_for_range)
        ylo, yhi = min(all_ys_for_range), max(all_ys_for_range)
        lo = min(xlo, ylo) / 3
        hi = max(xhi, yhi) * 3
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, label="y = x (pseudo-threshold)")
        ax.set_xlim(xlo / 2, xhi * 2)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("physical error rate (p)")
    ax.set_ylabel("per-round logical error rate")
    ax.set_title("qLDPC memory code calibration (BP+OSD) vs. qSIEVE Fig 8")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

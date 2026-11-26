"""
Generates the project's two "headline" figure sets into examples/figures/:

- figures/spacetime_volume.png: spacetime-volume speedup vs circuit size,
  one line per circuit family (from qldpc_scaling_experiment.py's output).
- figures/infidelity_<family>.png: one plot PER circuit family, circuit
  infidelity vs physical error rate (log-log), one line per size within
  that family (from stim_circuit_fidelity.py's output) -- e.g.
  infidelity_qrom.png has qrom/qrom_s1/qrom_s2/qrom_s3 together.

Splitting infidelity by family (rather than one combined plot across all
23 circuits) keeps each figure readable -- a single plot with every
circuit/size would be an illegible 23-line tangle.

Run from the repo root, in the MAIN venv, after both experiments:

    python examples/qldpc_scaling_experiment.py
    python examples/subthreshold_scaling.py
    python examples/stim_circuit_fidelity.py
    python examples/plot_figures.py

Needs matplotlib; if not already installed: pip install matplotlib
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from nasim.qasm import load_clifford_t_circuit

VOLUME_CSV = Path("examples/qldpc_scaling_results.csv")
FIDELITY_CSV = Path("examples/tier2_fidelity_sanity_check_results.csv")
FIGURES_DIR = Path("examples/figures")

FIXTURE_DIRS = ["tests/fixtures/cirq_ft_tgate", "tests/fixtures/qasmbench_tgate"]
_SIZE_SUFFIX = re.compile(r"_s\d+$")


def family_of(label: str) -> str:
    """'qrom_s2' -> 'qrom', 'adder_s1' -> 'adder', 'qrom' -> 'qrom'."""
    return _SIZE_SUFFIX.sub("", label)


def num_qubits_of(label: str) -> int:
    for fixture_dir in FIXTURE_DIRS:
        path = Path(fixture_dir) / f"{label}.qasm"
        if path.exists():
            n, _gates = load_clifford_t_circuit(str(path))
            return n
    raise FileNotFoundError(f"{label}.qasm not found in {FIXTURE_DIRS}")


def plot_spacetime_volume() -> None:
    if not VOLUME_CSV.exists():
        print(f"skipping spacetime_volume.png -- {VOLUME_CSV} not found "
              f"(run examples/qldpc_scaling_experiment.py first)")
        return

    by_family: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with open(VOLUME_CSV) as f:
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
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    out_path = FIGURES_DIR / "spacetime_volume.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_infidelity_by_family() -> None:
    if not FIDELITY_CSV.exists():
        print(f"skipping infidelity plots -- {FIDELITY_CSV} not found "
              f"(run examples/stim_circuit_fidelity.py first)")
        return

    # family -> circuit label -> [(p_ph, infidelity), ...]
    by_family: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    with open(FIDELITY_CSV) as f:
        for row in csv.DictReader(f):
            label = row["circuit"]
            infidelity = 1.0 - 10.0 ** float(row["log10_fidelity"])
            by_family[family_of(label)][label].append((float(row["p_ph"]), infidelity))

    for family, by_label in sorted(by_family.items()):
        fig, ax = plt.subplots(figsize=(7, 6))

        # order legend entries by qubit count (smallest first), not label text
        labels_by_size = sorted(by_label, key=num_qubits_of)
        for label in labels_by_size:
            points = sorted(by_label[label])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            n = num_qubits_of(label)
            ax.plot(xs, ys, marker="o", markersize=4, label=f"{label} ({n}q)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("physical error rate (p_ph)")
        ax.set_ylabel("circuit infidelity (1 - fidelity)")
        ax.set_title(f"{family}: circuit infidelity vs physical error rate")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

        out_path = FIGURES_DIR / f"infidelity_{family}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_spacetime_volume()
    plot_infidelity_by_family()


if __name__ == "__main__":
    main()

from __future__ import annotations

from matplotlib.patches import FancyBboxPatch, Rectangle

from nasim.circuit import Circuit, GateType

_WIRE_COLOR = "#2d3748"
_BOX_FACE = "#edf2f7"
_BOX_EDGE = "#2d3748"
_DOT_COLOR = "#2d3748"
_STAGE_BAND = "#f3effe"
_ONEQ_HIGHLIGHT = "#d69e2e"

_BOX_SIZE = 0.6
_DOT_SIZE = 110


def _draw_1q_box(ax, stage_idx: int, q: int, highlight: dict | None) -> None:
    if highlight is not None:
        face, edge, lw, z, text_color = highlight["color"], "#1a202c", 2.5, 4, "white"
    else:
        face, edge, lw, z, text_color = _BOX_FACE, _BOX_EDGE, 1.2, 2, _BOX_EDGE

    box = FancyBboxPatch(
        (stage_idx - _BOX_SIZE / 2, q - _BOX_SIZE / 2), _BOX_SIZE, _BOX_SIZE,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=face, edgecolor=edge, linewidth=lw, zorder=z,
    )
    ax.add_patch(box)
    ax.text(stage_idx, q, "1Q", ha="center", va="center", fontsize=9,
             family="serif", fontweight="bold", color=text_color, zorder=z + 1)


def _draw_2q_symbol(ax, stage_idx: int, qa: int, qb: int, highlight: dict | None) -> None:


    if highlight is not None:
        pending = highlight["kind"] == "2q_pending"
        color = highlight["color"]
        lw = 2.2 if pending else 3.2
        alpha = 0.6 if pending else 1.0
        dot_size = _DOT_SIZE * (0.85 if pending else 1.15)
    else:
        color, lw, alpha, dot_size = _DOT_COLOR, 1.4, 1.0, _DOT_SIZE

    ax.plot([stage_idx, stage_idx], [qa, qb], color=color, lw=lw, alpha=alpha, zorder=3, solid_capstyle="round")
    ax.scatter([stage_idx, stage_idx], [qa, qb], s=dot_size, color=color, alpha=alpha,
               zorder=4, edgecolors="white", linewidths=0.6)


def plot_circuit(circuit: Circuit, ax, *, current_stage: int | None = None, highlights: list[dict] | None = None) -> None:

    ax.clear()
    n_qubits = circuit.num_qubits
    n_stages = len(circuit.stages)
    last_stage = max(n_stages - 1, 0)

    ax.set_xlim(-1.1, last_stage + 0.9)
    ax.set_ylim(n_qubits - 0.4, -0.9)

    if current_stage is not None:
        ax.add_patch(Rectangle((current_stage - 0.5, -0.9), 1.0, n_qubits + 0.3,
                                facecolor=_STAGE_BAND, edgecolor="none", zorder=0))

    for q in range(n_qubits):
        ax.plot([-0.3, last_stage + 0.3], [q, q], color=_WIRE_COLOR, lw=1.2, zorder=1, solid_capstyle="round")
        ax.text(-0.5, q, rf"$q_{{{q}}}$", ha="right", va="center", fontsize=12, family="serif", zorder=2)

    h_1q_qubits: set[int] = set()
    h_2q: dict[frozenset, dict] = {}
    for h in highlights or []:
        if h["stage"] != current_stage:
            continue
        if h["kind"] == "1q":
            h_1q_qubits.update(h["qubits"])
        else:
            h_2q[frozenset(h["qubits"])] = h

    for stage_idx, stage in enumerate(circuit.stages):
        for gate in stage:
            if gate.type is GateType.ONE_QUBIT:
                (q,) = gate.qubits
                hl = {"color": _ONEQ_HIGHLIGHT} if (stage_idx == current_stage and q in h_1q_qubits) else None
                _draw_1q_box(ax, stage_idx, q, hl)
            else:
                qa, qb = gate.qubits
                hl = h_2q.get(frozenset((qa, qb))) if stage_idx == current_stage else None
                _draw_2q_symbol(ax, stage_idx, qa, qb, hl)

    ax.set_xticks(range(n_stages))
    ax.set_xticklabels([str(i) for i in range(n_stages)], fontsize=9, color="#718096")
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("circuit", fontsize=13, family="serif")
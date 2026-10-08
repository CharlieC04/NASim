from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from nasim.placement import PlacedPatch, Placement
from nasim.schedule import Schedule
from nasim.circuit import GateType
from nasim.viz.device import plot_placement
from nasim.viz.circuit import plot_circuit


_MERGE_COLOR = "#6b46c1"
_MERGE_PULSE = "#e9d8fd"
_ONEQ_COLOR = "#d69e2e"
_MOVE_COLOR = "#e53e3e"

_SURGERY_KINDS = ("merge_handover", "2q_substep", "mid_circuit_measure", "split_handover")


def group_ops_for_display(ops: list) -> list[list]:

    groups: list[list] = []
    i, n = 0, len(ops)
    while i < n:
        if ops[i].kind in _SURGERY_KINDS:
            group = [ops[i]]
            i += 1
            while i < n and ops[i].kind in ("2q_substep", "mid_circuit_measure"):
                group.append(ops[i])
                i += 1
            if i < n and ops[i].kind == "split_handover":
                group.append(ops[i])
                i += 1
            groups.append(group)
        else:
            groups.append([ops[i]])
            i += 1
    return groups


def _apply_op(current: dict, op) -> dict:

    if op.kind == "move_frame":
        current = dict(current)
        for move in op.detail["moves"]:
            prior = current[move.qubit]
            current[move.qubit] = PlacedPatch(qubit=move.qubit, patch=prior.patch, anchor=move.target)
    elif op.kind == "merge_handover":
        current = dict(current)
        qa, qb = op.detail["qubits"]
        merged_patch = op.detail["merged_patch"]
        anchor = op.detail["anchor"]
        origin_qubit = op.detail["origin_qubit"]
        other_qubit = qb if origin_qubit == qa else qa
        del current[other_qubit]
        current[origin_qubit] = PlacedPatch(qubit=origin_qubit, patch=merged_patch, anchor=anchor)
    elif op.kind == "split_handover":
        current = dict(current)
        origin_qubit = op.detail["origin_qubit"]
        del current[origin_qubit]
        for restored in op.detail["restored"]:
            current[restored.qubit] = restored
    return current


def snapshots_from_schedule(initial: Placement, groups: list[list]) -> list[Placement]:

    snapshots = [initial]
    current = {p.qubit: p for p in initial.patches}
    for group in groups:
        representative_idx = len(group) - 1
        for i, op in enumerate(group):
            if op.kind == "merge_handover":
                representative_idx = i
                break

        for i, op in enumerate(group):
            current = _apply_op(current, op)
            if i == representative_idx:
                snapshots.append(Placement(model=initial.model, patches=tuple(current[q] for q in sorted(current))))
    return snapshots


def _surgery_status(idx: int, total: int, group: list) -> str:
    merge_op = next((o for o in group if o.kind == "merge_handover"), None)
    split_op = next((o for o in group if o.kind == "split_handover"), None)
    substeps = [o for o in group if o.kind == "2q_substep"]
    measures = [o for o in group if o.kind == "mid_circuit_measure"]
    rounds = len(measures)
    total_cz = sum(o.detail["active_gates"] for o in substeps)
    total_meas = sum(o.detail["num_ancilla"] for o in measures)
    duration = sum(o.duration_us for o in group)

    if merge_op is not None:
        qubits, origin = merge_op.detail["qubits"], merge_op.detail["origin_qubit"]
    elif split_op is not None:
        qubits, origin = split_op.detail["qubits"], split_op.detail["origin_qubit"]
    else:
        origin = group[0].targets[0][0]
        qubits = [origin]

    return (
        f"frame {idx}/{total}: lattice surgery q{qubits} -> q{origin} "
        f"({rounds} rounds, {total_cz} CZ, {total_meas} meas, {duration:.1f}us)"
    )


class ScheduleViewer:

    def __init__(self, initial: Placement, schedule: Schedule, circuit):
        self.schedule = schedule
        self.circuit = circuit
        self.groups = group_ops_for_display(schedule.ops)
        self.snapshots = snapshots_from_schedule(initial, self.groups)
        self.frame_idx = 0
        self._xlim, self._ylim = self._global_bounds()

        self.fig, (self.ax, self.ax_circuit) = plt.subplots(
            1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.3, 1]}
        )
        plt.subplots_adjust(bottom=0.2)

        device_pos = self.ax.get_position()
        btn_w, btn_h, gap = 0.12, 0.06, 0.04
        cx = (device_pos.x0 + device_pos.x1) / 2
        ax_prev = self.fig.add_axes([cx - gap / 2 - btn_w, 0.05, btn_w, btn_h])
        ax_next = self.fig.add_axes([cx + gap / 2, 0.05, btn_w, btn_h])
        self.btn_prev = Button(ax_prev, "<< Prev")
        self.btn_next = Button(ax_next, "Next >>")
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._draw()

    def _global_bounds(self, margin_um: float = 10.0):
        xs, ys = [], []
        for snap in self.snapshots:
            for _, _, pos in snap.atoms():
                xs.append(pos.x)
                ys.append(pos.y)
        if not xs:
            return (-margin_um, margin_um), (-margin_um, margin_um)
        return (min(xs) - margin_um, max(xs) + margin_um), (min(ys) - margin_um, max(ys) + margin_um)

    def _on_prev(self, event) -> None:
        self.frame_idx = max(0, self.frame_idx - 1)
        self._draw()

    def _on_next(self, event) -> None:
        self.frame_idx = min(len(self.groups), self.frame_idx + 1)
        self._draw()

    def _on_key(self, event) -> None:
        if event.key == "right":
            self._on_next(event)
        elif event.key == "left":
            self._on_prev(event)

    def _draw_on(self, ax, ax_circuit, idx: int) -> None:
        ax.clear()
        placement = self.snapshots[idx]
        total = len(self.groups)
        stage_idx = None
        highlights: list[dict] = []

        if idx == 0:
            plot_placement(placement, ax=ax)
            status = "initial placement"
        else:
            group = self.groups[idx - 1]
            is_surgery = any(op.kind in _SURGERY_KINDS for op in group)
            stage_idx = group[0].detail.get("stage")

            if is_surgery:
                status = _surgery_status(idx, total, group)
                merge_op = next((o for o in group if o.kind == "merge_handover"), None)
                split_op = next((o for o in group if o.kind == "split_handover"), None)
                if merge_op is not None:
                    origin_qubit, qubits = merge_op.detail["origin_qubit"], merge_op.detail["qubits"]
                elif split_op is not None:
                    origin_qubit, qubits = split_op.detail["origin_qubit"], split_op.detail["qubits"]
                else:
                    origin_qubit = group[0].targets[0][0]
                    qubits = [origin_qubit]
                plot_placement(placement, ax=ax, targeted_qubits={origin_qubit}, pulse_active=True,
                                highlight_color=_MERGE_COLOR, pulse_color=_MERGE_PULSE)
                if len(qubits) == 2:
                    highlights.append({"stage": stage_idx, "qubits": qubits, "color": _MERGE_COLOR, "kind": "2q_active"})
            elif group[0].kind == "1q_stage":
                op = group[0]
                targeted = set(op.detail["qubits"])
                plot_placement(placement, ax=ax, targeted_qubits=targeted, pulse_active=True)
                status = f"frame {idx}/{total}: 1Q gate stage - targets={sorted(targeted)}"
                highlights.append({"stage": stage_idx, "qubits": sorted(targeted), "color": _ONEQ_COLOR, "kind": "1q"})
            elif group[0].kind == "move_frame":
                op = group[0]
                plot_placement(placement, ax=ax)
                for move in op.detail["moves"]:
                    ax.annotate(
                        "", xy=(move.target.x, move.target.y),
                        xytext=(move.source.x, move.source.y),
                        arrowprops=dict(arrowstyle="->", color=_MOVE_COLOR, lw=1.5, alpha=0.8),
                        zorder=5,
                    )
                status = f"frame {idx}/{total}: move frame - qubits={op.detail['qubits']}"
                if stage_idx is not None:
                    for gate in self.circuit.stages[stage_idx]:
                        if gate.type is GateType.TWO_QUBIT:
                            highlights.append({"stage": stage_idx, "qubits": gate.qubits, "color": _MOVE_COLOR, "kind": "2q_pending"})
            else:
                op = group[0]
                plot_placement(placement, ax=ax)
                status = f"frame {idx}/{total}: {op.kind}"

        ax.set_title(ax.get_title() + f"\n{status}")
        ax.set_xlim(*self._xlim)
        ax.set_ylim(*self._ylim)

        plot_circuit(self.circuit, ax_circuit, current_stage=stage_idx, highlights=highlights)

    def _draw(self) -> None:
        self._draw_on(self.ax, self.ax_circuit, self.frame_idx)
        self.fig.canvas.draw_idle()

    def save_frames(self, out_dir: str, dpi: int = 110) -> list[str]:
        import os

        os.makedirs(out_dir, exist_ok=True)
        paths = []
        for idx in range(len(self.snapshots)):
            fig, (ax, ax_circuit) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.3, 1]})
            self._draw_on(ax, ax_circuit, idx)
            path = os.path.join(out_dir, f"frame_{idx:03d}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        return paths

    def save_gif(self, path: str, fps: float = 1.0, dpi: int = 100) -> str:

        import io
        from PIL import Image

        frames = []
        for idx in range(len(self.snapshots)):
            fig, (ax, ax_circuit) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.3, 1]})
            self._draw_on(ax, ax_circuit, idx)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            plt.close(fig)
            buf.seek(0)
            frames.append(Image.open(buf).convert("RGB"))

        frames[0].save(
            path, format="GIF", save_all=True, append_images=frames[1:],
            duration=int(1000 / fps), loop=0,
        )
        return path

    def show(self) -> None:
        plt.show()
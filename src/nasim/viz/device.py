from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle

from nasim.device import Device
from nasim.placement import Placement
from nasim.surface_code import AtomRole

def _device_bounds(placement: Placement) -> tuple[float, float, float, float]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for placed in placement.patches:
        x0, y0, w, h = placement.bounding_box_um(placed)
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x0 + w)
        ys1.append(y0 + h)

    return min(xs0), min(ys0), max(xs1), max(ys1)

def plot_device(device: Device, show_blockade: bool = False, ax: Axes | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    xs = [p.x for p in device.sites]
    ys = [p.y for p in device.sites]
    ax.scatter(xs, ys, s=40, c="#2b6cb0", zorder=3, label="trap site")

    if show_blockade:
        for p in device.sites:
            ax.add_patch(
                Circle(
                    (p.x, p.y),
                    device.model.blockade_radius_um / 2,
                    fill=False,
                    linestyle="--",
                    linewidth=0.5,
                    color="#a0aec0",
                    zorder=1,
                )
            )

    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(
        f"{device.model.name} - {device.rows}x{device.cols}, pitch={device.pitch_um}um"
    )
    ax.legend(loc="upper right")
    return ax


_ROLE_COLOR = {
    AtomRole.DATA: "#2b6cb0",
    AtomRole.ANCILLA_X: "#dd6b20",
    AtomRole.ANCILLA_Z: "#38a169",
}
_ROLE_MARKER = {
    AtomRole.DATA: "o",
    AtomRole.ANCILLA_X: "s",
    AtomRole.ANCILLA_Z: "^",
}
_ROLE_LABEL = {
    AtomRole.DATA: "data",
    AtomRole.ANCILLA_X: "ancilla (X)",
    AtomRole.ANCILLA_Z: "ancilla (Z)",
}


def plot_placement(
    placement: Placement,
    ax: Axes | None = None,
    targeted_qubits: set[int] | None = None,
    pulse_active: bool = False,
    highlight_color: str = "#d69e2e",
    pulse_color: str = "#fefcbf",
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if pulse_active and placement.patches:
        x0, y0, x1, y1 = _device_bounds(placement)
        pad = 0.1 * max(x1 - x0, y1 - y0, 1.0)
        ax.add_patch(Rectangle(
            (x0 - pad, y0 - pad), (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad,
            facecolor=pulse_color, edgecolor="none", alpha=0.5, zorder=0,
            label="active pulse",
        ))

    targeted_qubits = targeted_qubits or set()
    for placed in placement.patches:
        is_target = placed.qubit in targeted_qubits
        x0, y0, w, h = placement.bounding_box_um(placed)
        ax.add_patch(Rectangle(
            (x0, y0), w, h, fill=False,
            linestyle="-" if is_target else "--",
            linewidth=2.5 if is_target else 1.0,
            edgecolor=highlight_color if is_target else "#718096",
            zorder=2 if is_target else 1,
        ))
        ax.annotate(
            f"q{placed.qubit}", (x0, y0 + h), xytext=(2, -2), textcoords="offset points",
            ha="left", va="top", fontsize=9, fontweight="bold",
            color=highlight_color if is_target else "#1a202c", zorder=4,
        )

    seen_roles = set()
    for qubit, atom, pos in placement.atoms():
        role = atom.role
        label = _ROLE_LABEL[role] if role not in seen_roles else None
        seen_roles.add(role)
        ax.scatter(
            pos.x, pos.y, s=45, c=_ROLE_COLOR[role], marker=_ROLE_MARKER[role],
            edgecolors="white", linewidths=0.5, zorder=3, label=label,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    d = placement.patches[0].patch.distance if placement.patches else "?"
    ax.set_title(
        f"{placement.model.name} - {len(placement.patches)} logical qubits, "
        f"d={d} rotated surface code"
    )
    ax.legend(loc="upper right")
    ax.margins(0.15)
    return ax
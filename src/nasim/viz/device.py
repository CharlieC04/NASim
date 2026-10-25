from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, Rectangle

from nasim.device import Device
from nasim.placement import Placement
from nasim.surface_code import AtomRole

_ZONE_FACE = {"storage": "#f7fafc", "entanglement": "#ebf8ff"}
_ZONE_EDGE = {"storage": "#cbd5e0", "entanglement": "#90cdf4"}

def _zone_bbox(zone) -> tuple[float, float, float, float]:

    x0s, y0s, x1s, y1s = [], [], [], []
    for slm in zone.slms:
        x0s.append(slm.origin.x)
        x1s.append(slm.origin.x + slm.cols * slm.pitch_x_um)
        y0s.append(slm.origin.y)
        y1s.append(slm.origin.y + slm.rows * slm.pitch_y_um)
    return min(min(x0s), min(x1s)), min(min(y0s), min(y1s)), max(max(x0s), max(x1s)), max(max(y0s), max(y1s))

def zone_bbox_padded(zone, *, pad_um: float = 3.0) -> tuple[float, float, float, float]:

    x0, y0, x1, y1 = _zone_bbox(zone)
    return x0 - pad_um, y0 - pad_um, x1 + pad_um, y1 + pad_um

def plot_zones(device: Device, ax: Axes) -> None:

    for zone in device.zones:
        x0, y0, x1, y1 = _zone_bbox(zone)
        face = _ZONE_FACE.get(zone.kind, "#f7fafc")
        edge = _ZONE_EDGE.get(zone.kind, "#cbd5e0")
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            facecolor=face, edgecolor=edge, linewidth=1.5, zorder=-2,
        ))
        ax.annotate(
            zone.kind, (x0, y0), xytext=(4, 4), textcoords="offset points",
            ha="left", va="bottom", fontsize=11, fontweight="bold", color=edge, zorder=-1,
        )

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
    device: Device | None = None
) -> Axes:

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if device is not None and device.zones:
        plot_zones(device, ax)

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
        is_factory = placed.qubit < 0
        is_target = placed.qubit in targeted_qubits
        x0, y0, w, h = placement.bounding_box_um(placed)
        default_edge = "#6b46c1" if is_factory else "#718096"
        ax.add_patch(Rectangle(
            (x0, y0), w, h, fill=False,
            linestyle="-" if is_target else "--",
            linewidth=2.5 if is_target else (1.6 if is_factory else 1.0),
            edgecolor=highlight_color if is_target else default_edge,
            zorder=2 if is_target else 1,
        ))
        label = f"F{-placed.qubit}" if is_factory else f"q{placed.qubit}"
        ax.annotate(
            label, (x0, y0 + h), xytext=(2, -2), textcoords="offset points",
            ha="left", va="top", fontsize=9, fontweight="bold",
            color=highlight_color if is_target else (default_edge if is_factory else "#1a202c"), zorder=4,
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
    num_logical = sum(1 for p in placement.patches if p.qubit >= 0)
    num_factories = len(placement.patches) - num_logical
    factory_suffix = f" + {num_factories} factory(ies)" if num_factories else ""
    ax.set_title(
        f"{placement.model.name} - {num_logical} logical qubits{factory_suffix}, "
        f"d={d} rotated surface code"
    )
    ax.legend(loc="upper right")
    ax.margins(0.15)
    return ax
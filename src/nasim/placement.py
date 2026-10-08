from __future__ import annotations

import math
import random
from dataclasses import dataclass

from nasim.circuit import Circuit
from nasim.device import Device
from nasim.geometry import Position
from nasim.model import Model
from nasim.surface_code import Patch


@dataclass(frozen=True)
class PlacedPatch:
    qubit: int
    patch: Patch
    anchor: Position

@dataclass(frozen=True)
class Placement:
    model: Model
    patches: tuple[PlacedPatch, ...]

    def atoms(self):
        for placed in self.patches:
            for atom, pos in placed.patch.embed(placed.anchor, self.model.gate_pair_dist_um):
                yield placed.qubit, atom, pos

    def bounding_box_um(self, placed: PlacedPatch) -> tuple[float, float, float, float]:
        unit_um = self.model.gate_pair_dist_um
        min_x, min_y, max_x, max_y = placed.patch.local_bounds
        x0 = placed.anchor.x + min_x * unit_um
        y0 = placed.anchor.y + min_y * unit_um
        return x0, y0, (max_x - min_x) * unit_um, (max_y - min_y) * unit_um

def _patch_radius(patch: Patch, unit_um: float) -> float:
    min_x, min_y, max_x, max_y = patch.local_bounds
    return 0.5 * unit_um * max(max_x - min_x, max_y - min_y)

def patch_separation_um(model: Model, distance: int) -> float:
    patch = Patch.rotated(distance)
    return 2 * _patch_radius(patch, model.gate_pair_dist_um) + model.gate_pair_dist_um

def piqasso_placement(
    circuit: Circuit,
    model: Model,
    device: Device,
    *,
    distance: int = 3,
    delta: float = 0.1,
    lambda_rep: float = 50.0,
    eta0: float = 2.0,
    eta_decay: float = 0.95,
    iterations: int = 200,
    seed: int = 0
):

    """
    Initial qubit placement based on the algorithm in the Piqasso paper: https://arxiv.org/pdf/2608.01316
    """

    n = circuit.num_qubits
    weights = circuit.interaction_weights(delta=delta)

    patch = Patch.rotated(distance)
    unit_um = model.gate_pair_dist_um
    d_star = patch_separation_um(model, distance)

    if device.min_site_spacing_um < d_star:
        raise ValueError("Pitch smaller than separation")
    if len(device.sites) < n:
        raise ValueError("Circuit too big")

    xs = [s.x for s in device.sites]
    ys = [s.y for s in device.sites]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    rng = random.Random(seed)
    positions = [
        [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)]
        for _ in range(n)
    ]

    eta = eta0
    for _ in range(iterations):
        forces = [[0.0, 0.0] for _ in range(n)]

        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                d = math.hypot(dx, dy) or 1e-6
                ux, uy = dx / d, dy / d

                w = weights.get((i, j), 0.0)
                if w:
                    spring = 2 * w * (d - d_star)
                    forces[i][0] -= spring * ux
                    forces[i][1] -= spring * uy
                    forces[j][0] += spring * ux
                    forces[j][1] += spring * uy

                rep = lambda_rep / (d * d)
                forces[i][0] += rep * ux
                forces[i][1] += rep * uy
                forces[j][0] -= rep * ux
                forces[j][1] -= rep * uy

        for i in range(n):
            positions[i][0] += eta * forces[i][0]
            positions[i][1] += eta * forces[i][1]
            positions[i][0] = max(x_min, min(x_max, positions[i][0]))
            positions[i][1] = max(y_min, min(y_max, positions[i][1]))

        eta *= eta_decay

    sites = [(s.x, s.y) for s in device.sites]
    degree = {i: sum(w for (a,b), w in weights.items() if i in (a,b)) for i in range(n)}
    order = sorted(range(n), key=lambda i: -degree[i])

    used = [False] * len(sites)
    anchors: list[tuple[float, float]] = [None] * n
    for i in order:
        px, py = positions[i]
        best_j, best_d = None, None
        for j, (sx, sy) in enumerate(sites):
            if used[j]: continue

            dd = (sx - px) ** 2 + (sy - py) ** 2
            if best_d is None or dd < best_d:
                best_d = dd
                best_j = j

        used[best_j] = True
        anchors[i] = sites[best_j]

    min_x, min_y, max_x, max_y = patch.local_bounds
    placed = tuple(
        PlacedPatch(
            qubit=i,
            patch=patch,
            anchor=Position(
                anchors[i][0] - min_x * unit_um,
                anchors[i][1] - min_y * unit_um,
                0.0
            )
        )
        for i in range(n)
    )

    return Placement(model=model, patches=placed)
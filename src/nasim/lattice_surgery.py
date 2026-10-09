from __future__ import annotations

import math
from enum import Enum

from nasim.placement import Placement
from nasim.surface_code import Edge, Patch
from nasim.geometry import Position

def facing_edges(placement: Placement, qubit_a: int, qubit_b: int) -> tuple[Edge, Edge] | None:

    """
    Check if patches are touching with matching extents
    """

    pa = next(p for p in placement.patches if p.qubit == qubit_a)
    pb = next(p for p in placement.patches if p.qubit == qubit_b)
    ax0, ay0, aw, ah = placement.bounding_box_um(pa)
    bx0, by0, bw, bh = placement.bounding_box_um(pb)

    if math.isclose(ax0 + aw, bx0) and math.isclose(ay0, by0) and math.isclose(ah, bh):
        return Edge.RIGHT, Edge.LEFT
    if math.isclose(bx0 + bw, ax0) and math.isclose(ay0, by0) and math.isclose(ah, bh):
        return Edge.LEFT, Edge.RIGHT
    if math.isclose(ay0 + ah, by0) and math.isclose(ax0, bx0) and math.isclose(aw, bw):
        return Edge.TOP, Edge.BOTTOM
    if math.isclose(by0 + bh, ay0) and math.isclose(ax0, bx0) and math.isclose(aw, bw):
        return Edge.BOTTOM, Edge.TOP

def check_alignment(placement: Placement, qubit_a: int, qubit_b: int) -> None:

    """
    Verify two patches positioned correctly for surgery
    """

    edges = facing_edges(placement, qubit_a, qubit_b)
    if edges is None:
        raise ValueError("Qubits not next to each other")

    pa = next(p for p in placement.patches if p.qubit == qubit_a)
    pb = next(p for p in placement.patches if p.qubit == qubit_b)
    edge_a, edge_b = edges
    role_a = pa.patch.boundary_role(edge_a)
    role_b = pb.patch.boundary_role(edge_b)
    if role_a != role_b:
        raise ValueError("Qubits have mismatched boundaries")

def merge_patches(placement: Placement, qubit_a: int, qubit_b: int) -> tuple[Patch, Position, int]:

    """
    Build the merged patch for two aligned, adjacent qubits. Returns
    (merged_patch, anchor, origin_qubit), where origin_qubit is whichever
    of qubit_a/qubit_b sits at the merged patch's own local origin
    """

    check_alignment(placement, qubit_a, qubit_b)
    edge_a, edge_b = facing_edges(placement, qubit_a, qubit_b)
    pa = next(p for p in placement.patches if p.qubit == qubit_a)
    pb = next(p for p in placement.patches if p.qubit == qubit_b)

    if edge_a is Edge.RIGHT:
        origin, horizontal = pa, True
    elif edge_a is Edge.LEFT:
        origin, horizontal = pb, True
    elif edge_a is Edge.TOP:
        origin, horizontal = pa, False
    else:
        origin, horizontal = pb, False

    if pa.patch.distance != pb.patch.distance:
        raise ValueError("Patches have different d")

    d = origin.patch.distance

    if horizontal:
        merged = Patch._build_rectangle(2 * d, d, code_distance=d)
    else:
        merged = Patch._build_rectangle(d, 2 * d, code_distance=d)

    return merged, origin.anchor, origin.qubit
from __future__ import annotations

import stim
import tqec
from tqec.computation.block_graph import BlockGraph
from tqec.computation.cube import ZXCube
from tqec.utils.noise_model import NoiseModel
from tqec.utils.position import Position3D

_PORT_KIND = {"Z": "ZXZ", "X": "XZX"}

def merge_split_block_graph(basis: str = "Z") -> BlockGraph:

    """
    Two rotated-surface-code patches, merged via a spatial lattice-surgery
    pipe for one time slab then split back apart.
    """

    kind_str = _PORT_KIND[basis]
    port_kind = ZXCube.from_str(kind_str)

    g = BlockGraph("Merge-Split")
    g.add_cube(Position3D(0, 0, 0), "P", "In_A")
    g.add_cube(Position3D(0, 0, 1), kind_str, "A")
    g.add_cube(Position3D(1, 0, 1), kind_str, "B")
    g.add_cube(Position3D(0, 0, 2), "P", "Out_A")
    g.add_cube(Position3D(1, 0, 2), "P", "Out_B")
    g.add_pipe(Position3D(0, 0, 0), Position3D(0, 0, 1))  # In_A -> A
    g.add_pipe(Position3D(0, 0, 1), Position3D(1, 0, 1))  # merge (A -> B)
    g.add_pipe(Position3D(0, 0, 1), Position3D(0, 0, 2))  # A -> Out_A
    g.add_pipe(Position3D(1, 0, 1), Position3D(1, 0, 2))  # B -> Out_B
    g.fill_ports(port_kind)
    return g

def merge_split_circuit(*, k: int, p: float, basis: str = "Z") -> stim.Circuit:

    g = merge_split_block_graph(basis=basis)
    compiled = tqec.compile_block_graph(g)
    return compiled.generate_stim_circuit(k=k, noise_model=NoiseModel.uniform_depolarizing(p))
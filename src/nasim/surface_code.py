from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nasim.geometry import Position

class AtomRole(Enum):

    DATA = "data"
    ANCILLA_X = "ancilla_x"
    ANCILLA_Z = "ancilla_z"

class Edge(Enum):

    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"

@dataclass(frozen=True)
class PatchAtom:
    role: AtomRole
    local: tuple[int, int]

@dataclass(frozen=True)
class Patch:

    distance: int
    atoms: tuple[PatchAtom, ...]

    @property
    def num_data(self) -> int:
        return sum(1 for a in self.atoms if a.role == AtomRole.DATA)

    @property
    def num_ancilla(self) -> int:
        return sum(1 for a in self.atoms if a.role != AtomRole.DATA)

    @property
    def local_bounds(self) -> tuple[int, int, int, int]:
        xs = [a.local[0] for a in self.atoms]
        ys = [a.local[1] for a in self.atoms]
        return min(xs), min(ys), max(xs), max(ys)

    def radius_um(self, unit_um: float) -> float:
        min_x, min_y, max_x, max_y = self.local_bounds
        return 0.5 * unit_um * max(max_x - min_x, max_y - min_y)

    @classmethod
    def rotated(cls, distance: int) -> Patch:

        if distance < 3 or distance % 2 == 0:
            raise ValueError("Invalid surface code dist")

        return cls._build_rectangle(distance, distance, code_distance=distance)
        

    @classmethod
    def _build_rectangle(cls, dx: int, dy: int, *, code_distance: int) -> Patch:

        atoms: list[PatchAtom] = []
        
        for row in range(dy):
            for col in range(dx):
                atoms.append(PatchAtom(AtomRole.DATA, (2 * col, 2 * row)))

        for row in range(dy-1):
            for col in range(dx-1):
                is_x = (row + col) % 2 == 0
                role = AtomRole.ANCILLA_X if is_x else AtomRole.ANCILLA_Z
                atoms.append(PatchAtom(role, (2 * col + 1, 2 * row + 1)))

        for col in range(dx-1):
            x = 2 * col + 1
            if col % 2 == 1:
                atoms.append(PatchAtom(AtomRole.ANCILLA_Z, (x, -1)))
            else:
                atoms.append(PatchAtom(AtomRole.ANCILLA_Z, (x, 2*dy - 1)))

        for row in range(dy-1):
            y = 2 * row + 1
            if row % 2 == 0:
                atoms.append(PatchAtom(AtomRole.ANCILLA_X, (-1, y)))
            else:
                atoms.append(PatchAtom(AtomRole.ANCILLA_X, (2*dx - 1, y)))

        return cls(distance=code_distance, atoms=tuple(atoms))

    def embed(self, anchor: Position, unit_um: float) -> list[tuple[PatchAtom, Position]]:
        
        """
            Absolute positions for each atom in patch, given position of local (0,0)
        """

        return [
            (
                atom,
                Position(
                    anchor.x + atom.local[0] * unit_um,
                    anchor.y + atom.local[1] * unit_um,
                    anchor.z
                )
            )
            for atom in self.atoms
        ]

    def boundary_role(self, edge: Edge) -> AtomRole:

        if edge in (Edge.TOP, Edge.BOTTOM): return AtomRole.ANCILLA_Z
        return AtomRole.ANCILLA_X

def _ancilla_weight_counts(patch: Patch) -> tuple[int, int]:

    min_x, min_y, max_x, max_y = patch.local_bounds
    interior = 0
    boundary = 0
    for a in patch.atoms:
        if a.role == AtomRole.DATA: continue

        x,y = a.local
        if x in (min_x, max_x) or y in (min_y, max_y): boundary += 1
        else: interior += 1

    return interior, boundary
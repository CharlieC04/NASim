from __future__ import annotations

from dataclasses import dataclass, field
import math

from nasim.geometry import Position
from nasim.model import Model

@dataclass(frozen=True)
class SLM:

    """
    One SLM trap array (r x c lattice)
    """

    rows: int
    cols: int
    pitch_x_um: float
    pitch_y_um: float
    origin: Position = Position(0.0, 0.0, 0.0)

    @property
    def sites(self) -> tuple[Position, ...]:
        return tuple(
            Position(
                self.origin.x + col * self.pitch_x_um,
                self.origin.y + row * self.pitch_y_um,
                self.origin.z
            )
            for row in range(self.rows)
            for col in range(self.cols)
        )

@dataclass(frozen=True)
class Device:
    """
    A 2D grid of trap sites for a given machine model
    """

    model: Model
    slms: tuple[SLM, ...]
    sites: tuple[Position, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        seen: dict[tuple[float, float, float], Position] = {}
        for slm in self.slms:
            for s in slm.sites:
                seen[(s.x, s.y, s.z)] = s
        object.__setattr__(self, "sites", tuple(seen.values()))

    def site_id(self, row: int, col: int) -> int:
        """Only meaningful for a single uniform SLM block."""
        return row * self.slms[0].cols + col

    @classmethod
    def uniform(cls, model: Model, rows: int, cols: int, pitch_um: float) -> Device:

        """
        Constructor for single-SLM, square-pitch
        """

        if pitch_um < model.gate_pair_dist_um:
            raise ValueError("2Q Gates not Possible")

        return cls(
            model=model,
            slms=(SLM(rows=rows, cols=cols, pitch_x_um=pitch_um, pitch_y_um=pitch_um),)
        )

    @property
    def min_site_spacing_um(self) -> float:

        sites = self.sites
        best = math.inf
        for i in range(len(sites)):
            for j in range(i + 1, len(sites)):
                d = math.dist((sites[i].x, sites[i].y), (sites[j].x, sites[j].y))
                best = min(d, best)

        return best
from __future__ import annotations

from dataclasses import dataclass, field
import math

from nasim.geometry import Position
from nasim.model import Model

@dataclass(frozen=True)
class SLM:

    """
    SLM trap array (r x c lattice)
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
class Zone:

    """
    Each zone has ONE SLM for now
    """

    name: str
    kind: str
    slm: SLM

    @property
    def sites(self) -> tuple[Position, ...]:
        return self.slm.sites

    def contains(self, pos: Position) -> bool:

        x0 = self.slm.origin.x
        x1 = self.slm.origin.x + self.slm.cols * self.slm.pitch_x_um
        y0 = self.slm.origin.y
        y1 = self.slm.origin.y + self.slm.rows * self.slm.pitch_y_um

        return min(x0, x1) <= pos.x <= max(x0, x1) and min(y0, y1) <= pos.y <= max(y0, y1)

def _slm_bbox(slm: SLM) -> tuple[float, float, float, float]:

    x0 = slm.origin.x
    x1 = slm.origin.x + (slm.cols - 1) * slm.pitch_x_um
    y0 = slm.origin.y
    y1 = slm.origin.y + (slm.rows - 1) * slm.pitch_y_um

    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

def _slms_overlap(a: SLM, b: SLM) -> bool:
    ax0, ay0, ax1, ay1 = _slm_bbox(a)
    bx0, by0, bx1, by1 = _slm_bbox(b)
    return ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1

@dataclass(frozen=True)
class Device:
    """
    One or more SLM trap-site grids merged into one addressable site set,
    for a given machine Model. Build with Device.uniform (one grid, no
    zones) or Device.zoned (separate storage/entanglement grids).
    """

    model: Model
    slms: tuple[SLM, ...]
    zones: tuple[Zone, ...] = ()
    sites: tuple[Position, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        seen: dict[tuple[float, float, float], Position] = {}
        for slm in self.slms:
            for s in slm.sites:
                seen[(s.x, s.y, s.z)] = s
        object.__setattr__(self, "sites", tuple(seen.values()))

    def site_id(self, row: int, col: int) -> int:
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

    @classmethod
    def zoned(cls, model: Model, *, storage_rows: int, storage_cols: int, storage_pitch_um: float,
              entanglement_rows: int, entanglement_cols: int, entanglement_pitch: float, gap_um: float = 10.0):

        """Build a two-zone device: a storage grid with the entanglement
        grid placed directly above it, separated by gap_um."""

        storage = SLM(rows=storage_rows, cols=storage_cols, pitch_x_um=storage_pitch_um, pitch_y_um=storage_pitch_um)
        storage_height = (storage_rows - 1) * storage_pitch_um
        entanglement = SLM(rows=entanglement_rows, cols=entanglement_cols, pitch_x_um=entanglement_pitch, pitch_y_um=entanglement_pitch,
                           origin=Position(0.0, storage_height + gap_um, 0.0))

        if _slms_overlap(storage, entanglement):
            raise ValueError("Zones overlap")

        zones = (
            Zone(name="storage", kind="storage", slm=storage),
            Zone(name="entanglement", kind="entanglement", slm=entanglement)
        )

        return cls(model=model, slms=(storage, entanglement), zones=zones)

    def zone_of(self, pos: Position) -> Zone | None:

        for zone in self.zones:
            if zone.contains(pos):
                return zone

    @property
    def min_site_spacing_um(self) -> float:

        sites = self.sites
        best = math.inf
        for i in range(len(sites)):
            for j in range(i + 1, len(sites)):
                d = math.dist((sites[i].x, sites[i].y), (sites[j].x, sites[j].y))
                best = min(d, best)

        return best
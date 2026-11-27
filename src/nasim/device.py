from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math

from nasim.config.loader import load_config
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

def _slm_bbox(slm: SLM) -> tuple[float, float, float, float]:

    x0 = slm.origin.x
    x1 = slm.origin.x + (slm.cols - 1) * slm.pitch_x_um
    y0 = slm.origin.y
    y1 = slm.origin.y + (slm.rows - 1) * slm.pitch_y_um

    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

def _slm_padded_bbox(slm: SLM) -> tuple[float, float, float, float]:

    x0 = slm.origin.x
    x1 = slm.origin.x + slm.cols * slm.pitch_x_um
    y0 = slm.origin.y
    y1 = slm.origin.y + slm.rows * slm.pitch_y_um

    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

def _slms_overlap(a: SLM, b: SLM) -> bool:
    if a.origin.z != b.origin.z: return False
    ax0, ay0, ax1, ay1 = _slm_bbox(a)
    bx0, by0, bx1, by1 = _slm_bbox(b)
    return ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1

@dataclass(frozen=True)
class Zone:

    """
    Each zone has ONE SLM for now
    """

    name: str
    kind: str
    slms: tuple[SLM, ...]
    num_aods: int = 1

    @property
    def sites(self) -> tuple[Position, ...]:
        seen: dict[tuple[float, float, float], Position] = {}
        for slm in self.slms:
            for s in slm.sites:
                seen[(s.x, s.y, s.z)] = s

        return tuple(seen.values())

    def contains(self, pos: Position) -> bool:

        for slm in self.slms:
            if slm.origin.z != pos.z: continue
            x0, y0, x1, y1 = _slm_padded_bbox(slm)
            if x0 <= pos.x <= x1 and y0 <= pos.y <= y1: return True
        return False

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
              entanglement_rows: int, entanglement_cols: int, entanglement_pitch: float, gap_um: float = 10.0,
              storage_num_aods: int = 1, entanglement_num_aods: int = 1,
              factory_rows: int | None = None, factory_cols: int | None = None,
              factory_pitch_um: float | None = None, factory_num_aods: int = 1,
              factory_gap_um: float | None = None):

        """Build a two-zone device: a storage grid with the entanglement
        grid placed directly above it, separated by gap_um."""

        storage = SLM(rows=storage_rows, cols=storage_cols, pitch_x_um=storage_pitch_um, pitch_y_um=storage_pitch_um)
        storage_height = storage_rows * storage_pitch_um
        entanglement = SLM(rows=entanglement_rows, cols=entanglement_cols, pitch_x_um=entanglement_pitch, pitch_y_um=entanglement_pitch,
                           origin=Position(0.0, storage_height + gap_um, 0.0))

        if _slms_overlap(storage, entanglement):
            raise ValueError("Zones overlap")

        slms = [storage, entanglement]
        zones = [
            Zone(name="storage", kind="storage", slms=(storage,), num_aods=storage_num_aods),
            Zone(name="entanglement", kind="entanglement", slms=(entanglement,), num_aods=entanglement_num_aods)
        ]

        if factory_rows is not None:

            if factory_cols is None or factory_pitch_um is None:
                raise ValueError("missing factory param")

            f_gap_um = gap_um if factory_gap_um is None else factory_gap_um
            entanglement_height = entanglement_rows * entanglement_pitch
            factory = SLM(rows=factory_rows, cols=factory_cols, pitch_x_um=factory_pitch_um,
                          pitch_y_um=factory_pitch_um, origin=Position(0.0, storage_height + 
                                                                       gap_um + entanglement_height + f_gap_um, 0.0))
            if _slms_overlap(entanglement, factory) or _slms_overlap(factory, storage):
                raise ValueError("zones overlap")

            slms.append(factory)
            zones.append(Zone(name="factory", kind="factory", slms=(factory,), num_aods=factory_num_aods))

        return cls(model=model, slms=tuple(slms), zones=tuple(zones))

    @classmethod
    def layered(cls, model: Model, *, memory_rows: int, memory_cols: int, memory_pitch_um: float,
                compute_rows: int, compute_cols: int, compute_pitch_um: float,
                factory_rows: int, factory_cols: int, factory_pitch_um: float,
                layer_spacing_um: float | None = None, memory_num_aods: int = 1,
                compute_num_aods: int = 1, factory_num_aods: int = 1) -> Device:

        spacing = model.layer_spacing_um if layer_spacing_um is None else layer_spacing_um
        if spacing < model.axial_clearance_um:
            raise ValueError("layer clearance too small")

        memory = SLM(rows=memory_rows, cols=memory_cols, pitch_x_um=memory_pitch_um,
                     pitch_y_um=memory_pitch_um, origin=Position(0.0, 0.0, 0.0))
        compute = SLM(rows=compute_rows, cols=compute_cols, pitch_x_um=compute_pitch_um,
                      pitch_y_um=compute_pitch_um, origin=Position(0.0, 0.0, spacing))
        factory = SLM(rows=factory_rows, cols=factory_cols, pitch_x_um=factory_pitch_um,
                      pitch_y_um=factory_pitch_um, origin=Position(0.0, 0.0, 2 * spacing))

        zones = (
            Zone(name="storage", kind="storage", slms=(memory,), num_aods=memory_num_aods),
            Zone(name="entanglement", kind="entanglement", slms=(compute,), num_aods=compute_num_aods),
            Zone(name="factory", kind="factory", slms=(factory,), num_aods=factory_num_aods),
        )

        return cls(model=model, slms=(memory, compute, factory), zones=zones)

    @classmethod
    def from_yaml(cls, path: str | Path, model: Model) -> Device:

        data = load_config(path)
        architecture = data.get("architecture")
        if architecture is None:
            raise ValueError("Config missing arch section")

        def build_zone(kind: str, zone_cfg: dict) -> Zone:
            slms = tuple(
                SLM(
                    rows=slm_cfg["rows"], cols=slm_cfg["cols"],
                    pitch_x_um=slm_cfg["pitch_x_um"], pitch_y_um=slm_cfg["pitch_y_um"],
                    origin=Position(slm_cfg["origin_um"][0], slm_cfg["origin_um"][1], 0.0)
                )
                for slm_cfg in zone_cfg["slms"]
            )
            name = f"{kind}_{zone_cfg["zone_id"]}"

            return Zone(name=name, kind=kind, slms=slms, num_aods=zone_cfg.get("num_aods", 1))

        zones = tuple(
            build_zone(kind, zone_cfg)
            for kind, key in (("storage", "storage_zones"), ("entanglement", "entanglement_zones"))
            for zone_cfg in architecture.get(key, [])
        )

        all_slms = tuple(slm for zone in zones for slm in zone.slms)
        for i in range(len(all_slms)):
            for j in range(i + 1, len(all_slms)):
                if _slms_overlap(all_slms[i], all_slms[j]):
                    raise ValueError("SLMs overlap")

        return cls(model=model, slms=all_slms, zones=zones)

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
                if sites[i].z != sites[j].z: continue
                d = math.dist((sites[i].x, sites[i].y), (sites[j].x, sites[j].y))
                best = min(d, best)

        return best

from pathlib import Path

import pytest

from nasim.device import Device
from nasim.geometry import Position
from nasim.model import Model

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"


@pytest.fixture
def model() -> Model:
    return Model.from_yaml(PRESET)


def test_site_count(model):
    device = Device.uniform(model=model, rows=3, cols=4, pitch_um=10.0)
    assert len(device.sites) == 12


def test_sites_are_pitch_apart(model):
    device = Device.uniform(model=model, rows=2, cols=2, pitch_um=10.0)
    assert device.sites[0] == (0.0, 0.0, 0.0)
    assert device.sites[1] == (10.0, 0.0, 0.0)
    assert device.sites[device.site_id(1, 0)] == (0.0, 10.0, 0.0)


def test_all_sites_have_z_zero(model):
    device = Device.uniform(model=model, rows=2, cols=2, pitch_um=10.0)
    assert all(p.z == 0.0 for p in device.sites)


def test_site_id_matches_row_major_order(model):
    device = Device.uniform(model=model, rows=3, cols=3, pitch_um=10.0)
    assert device.site_id(0, 0) == 0
    assert device.site_id(1, 2) == 5
    assert device.site_id(2, 2) == 8


def test_rejects_pitch_below_gate_pair_distance(model):
    with pytest.raises(ValueError):
        Device.uniform(model=model, rows=2, cols=2, pitch_um=1.0)


def test_min_site_spacing_matches_pitch(model):
    device = Device.uniform(model=model, rows=3, cols=3, pitch_um=10.0)
    assert device.min_site_spacing_um == pytest.approx(10.0)


def test_multi_slm_device_merges_and_dedups_sites(model):
    from nasim.device import SLM
    from nasim.geometry import Position

    slm1 = SLM(rows=2, cols=2, pitch_x_um=10.0, pitch_y_um=10.0, origin=Position(0.0, 0.0, 0.0))
    slm2 = SLM(rows=2, cols=2, pitch_x_um=10.0, pitch_y_um=10.0, origin=Position(0.0, 0.0, 0.0))
    device = Device(model=model, slms=(slm1, slm2))
    assert len(device.sites) == 4  # fully overlapping blocks dedup


def test_uniform_device_has_no_zones(model):
    device = Device.uniform(model=model, rows=3, cols=3, pitch_um=10.0)
    assert device.zones == ()


def test_zoned_device_creates_storage_and_entanglement_zones(model):
    device = Device.zoned(
        model=model,
        storage_rows=10, storage_cols=10, storage_pitch_um=10.0,
        entanglement_rows=4, entanglement_cols=4, entanglement_pitch=10.0,
        gap_um=10.0,
    )
    assert {z.kind for z in device.zones} == {"storage", "entanglement"}
    assert {z.name for z in device.zones} == {"storage", "entanglement"}


def test_zoned_device_zones_do_not_overlap(model):
    from nasim.device import _slms_overlap

    device = Device.zoned(
        model=model,
        storage_rows=10, storage_cols=10, storage_pitch_um=10.0,
        entanglement_rows=4, entanglement_cols=4, entanglement_pitch=10.0,
        gap_um=10.0,
    )
    storage = next(z for z in device.zones if z.kind == "storage")
    entanglement = next(z for z in device.zones if z.kind == "entanglement")
    assert not _slms_overlap(storage.slm, entanglement.slm)


def test_zoned_device_merges_sites_from_both_zones(model):
    device = Device.zoned(
        model=model,
        storage_rows=10, storage_cols=10, storage_pitch_um=10.0,
        entanglement_rows=4, entanglement_cols=4, entanglement_pitch=10.0,
        gap_um=10.0,
    )
    assert len(device.sites) == 10 * 10 + 4 * 4


def test_zone_of_classifies_a_storage_position(model):
    device = Device.zoned(
        model=model,
        storage_rows=10, storage_cols=10, storage_pitch_um=10.0,
        entanglement_rows=4, entanglement_cols=4, entanglement_pitch=10.0,
        gap_um=10.0,
    )
    zone = device.zone_of(Position(50.0, 50.0, 0.0))
    assert zone is not None
    assert zone.kind == "storage"


def test_zone_of_classifies_an_entanglement_position(model):
    # gap_um is deliberately larger than pitch here: each site is the
    # corner of a pitch-wide cell (not a bare point), so a zone's box
    # extends a full `rows`/`cols` * pitch, not (rows-1)/(cols-1) * pitch -
    # with gap_um == pitch the two zones' widened boxes would touch with
    # no real gap left to test.
    device = Device.zoned(
        model=model,
        storage_rows=10, storage_cols=10, storage_pitch_um=10.0,
        entanglement_rows=4, entanglement_cols=4, entanglement_pitch=10.0,
        gap_um=20.0,
    )
    # storage's cell-based box spans y in [0, 100], but Device.zoned places
    # entanglement at y = (storage_rows-1)*pitch + gap_um = 90 + 20 = 110
    zone = device.zone_of(Position(0.0, 125.0, 0.0))
    assert zone is not None
    assert zone.kind == "entanglement"


def test_zone_of_returns_none_in_the_gap_between_zones(model):
    device = Device.zoned(
        model=model,
        storage_rows=10, storage_cols=10, storage_pitch_um=10.0,
        entanglement_rows=4, entanglement_cols=4, entanglement_pitch=10.0,
        gap_um=20.0,
    )
    # storage's cell-based box ends at y=100, entanglement starts at
    # y=90+gap_um=110: y=105 is in the gap
    assert device.zone_of(Position(0.0, 105.0, 0.0)) is None


def test_zone_contains_the_full_cell_at_the_last_site_not_just_the_point(model):
    # Each site is the corner of a pitch-wide cell (matching
    # patch_separation_um, which sizes the pitch to hold exactly one
    # patch), not a bare point - so a zone's box must extend a full
    # `rows`/`cols` * pitch beyond the origin, not (rows-1)/(cols-1) * pitch.
    # A patch anchored at the LAST site (after the piqasso-style
    # anchor correction, _site_anchor in schedule.py) has its local
    # origin sitting model.gate_pair_dist_um beyond that site - this
    # checks the zone's box has room for that, which only (rows)*pitch
    # (not (rows-1)*pitch) guarantees in general.
    device = Device.zoned(
        model=model,
        storage_rows=4, storage_cols=4, storage_pitch_um=10.0,
        entanglement_rows=2, entanglement_cols=2, entanglement_pitch=10.0,
        gap_um=10.0,
    )
    storage_zone = next(z for z in device.zones if z.kind == "storage")
    last_site = storage_zone.sites[-1]  # row=3, col=3 -> (30.0, 30.0)
    assert last_site == Position(30.0, 30.0, 0.0)
    just_beyond = Position(last_site.x + model.gate_pair_dist_um, last_site.y + model.gate_pair_dist_um, 0.0)
    assert storage_zone.contains(just_beyond)


def test_zoned_rejects_overlapping_zones(model):
    with pytest.raises(ValueError):
        Device.zoned(
            model=model,
            storage_rows=10, storage_cols=10, storage_pitch_um=10.0,
            entanglement_rows=4, entanglement_cols=4, entanglement_pitch=10.0,
            gap_um=-50.0,  # forces overlap
        )
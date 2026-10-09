from pathlib import Path

from nasim.circuit import Circuit, one_qubit, two_qubit
from nasim.device import Device
from nasim.geometry import Position
from nasim.model import Model
from nasim.placement import PlacedPatch, Placement, patch_separation_um
from nasim.schedule import circuit_fidelity, simulate_circuit
from nasim.surface_code import Patch
from nasim.viz import ScheduleViewer

PRESET = Path(__file__).resolve().parents[1] / "src" / "nasim" / "config" / "presets" / "2d.yaml"


def site_anchor(site: Position, patch: Patch, unit_um: float) -> Position:

    min_x, min_y, _, _ = patch.local_bounds
    return Position(site.x - min_x * unit_um, site.y - min_y * unit_um, site.z)


def main() -> None:
    model = Model.from_yaml(PRESET)
    pitch = patch_separation_um(model, distance=3)

    device = Device.zoned(
        model=model,
        storage_rows=4, storage_cols=4, storage_pitch_um=pitch,
        entanglement_rows=2, entanglement_cols=2, entanglement_pitch=pitch,
        gap_um=pitch,
    )
    storage_zone = next(z for z in device.zones if z.kind == "storage")

    patch = Patch.rotated(3)
    unit_um = model.gate_pair_dist_um
    sites = storage_zone.sites
    placement = Placement(model=model, patches=(
        PlacedPatch(0, patch, site_anchor(sites[0], patch, unit_um)),
        PlacedPatch(1, patch, site_anchor(sites[3], patch, unit_um)),
        PlacedPatch(2, patch, site_anchor(sites[12], patch, unit_um)),
        PlacedPatch(3, patch, site_anchor(sites[15], patch, unit_um)),
    ))

    circuit = Circuit.from_stages(4, [
        [one_qubit(0), one_qubit(1), one_qubit(2), one_qubit(3)],
        [two_qubit(0, 1)],
        [two_qubit(1, 2)],
        [one_qubit(3)],
    ])

    schedule, final_placement = simulate_circuit(circuit, placement, model, device=device)

    print(f"{len(schedule.ops)} raw ops")
    print(f"Nxtalk={schedule.Nxtalk} Nh={schedule.Nh} clock={schedule.clock_us:.2f}us")
    print(f"circuit fidelity: {circuit_fidelity(schedule):.6g}")

    ent_zone = next(z for z in device.zones if z.kind == "entanglement")
    for p in final_placement.patches:
        loc = "entanglement" if ent_zone.contains(p.anchor) else "storage" if storage_zone.contains(p.anchor) else "?"
        print(f"  q{p.qubit}: {loc}")

    viewer = ScheduleViewer(placement, schedule, circuit, device=device)
    saved = viewer.save_frames("zone_demo_frames")
    print(f"saved {len(saved)} frame(s) to zone_demo_frames/")
    viewer.save_gif("zone_demo.gif", fps=1.0)
    viewer.show()


if __name__ == "__main__":
    main()
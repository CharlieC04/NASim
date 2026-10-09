import pytest

from nasim.geometry import Position
from nasim.surface_code import AtomRole, Patch


@pytest.mark.parametrize("d", [3, 5, 7])
def test_physical_qubit_counts(d):
    patch = Patch.rotated(d)
    data = [a for a in patch.atoms if a.role == AtomRole.DATA]
    ancilla = [a for a in patch.atoms if a.role != AtomRole.DATA]
    assert len(data) == d * d
    assert len(ancilla) == d * d - 1
    assert len(patch.atoms) == 2 * d * d - 1


@pytest.mark.parametrize("d", [3, 5, 7])
def test_num_data_and_num_ancilla_match_actual_atom_counts(d):
    patch = Patch.rotated(d)
    assert patch.num_data == d * d
    assert patch.num_ancilla == d * d - 1


def test_num_data_and_num_ancilla_correct_for_a_merged_rectangular_patch():
    # a square patch's distance**2 formula is wrong once merging produces a
    # rectangular patch - num_data/num_ancilla must count the actual atoms
    merged = Patch._build_rectangle(6, 3, code_distance=3)
    data = [a for a in merged.atoms if a.role == AtomRole.DATA]
    ancilla = [a for a in merged.atoms if a.role != AtomRole.DATA]
    assert merged.num_data == len(data) == 18
    assert merged.num_ancilla == len(ancilla) == 17


def test_rejects_even_distance():
    with pytest.raises(ValueError):
        Patch.rotated(4)


def test_rejects_distance_below_3():
    with pytest.raises(ValueError):
        Patch.rotated(1)


def test_no_duplicate_local_positions():
    patch = Patch.rotated(5)
    locals_ = [a.local for a in patch.atoms]
    assert len(locals_) == len(set(locals_))


def test_data_qubits_form_d_by_d_grid():
    d = 3
    patch = Patch.rotated(d)
    data_locals = {a.local for a in patch.atoms if a.role == AtomRole.DATA}
    expected = {(2 * c, 2 * r) for r in range(d) for c in range(d)}
    assert data_locals == expected


def test_interior_ancillas_checkerboard_colored():
    patch = Patch.rotated(3)
    by_local = {a.local: a.role for a in patch.atoms}
    assert by_local[(1, 1)] == AtomRole.ANCILLA_X
    assert by_local[(3, 1)] == AtomRole.ANCILLA_Z
    assert by_local[(1, 3)] == AtomRole.ANCILLA_Z
    assert by_local[(3, 3)] == AtomRole.ANCILLA_X


def test_boundary_ancillas_d3():
    patch = Patch.rotated(3)
    boundary_locals = {a.local for a in patch.atoms if -1 in a.local or 5 in a.local}
    assert boundary_locals == {(3, -1), (1, 5), (-1, 1), (5, 3)}


def test_embed_places_anchor_at_first_data_qubit():
    patch = Patch.rotated(3)
    embedded = dict(patch.embed(Position(10.0, 20.0, 0.0), unit_um=2.0))
    data_atom = next(a for a in patch.atoms if a.local == (0, 0))
    assert embedded[data_atom] == Position(10.0, 20.0, 0.0)
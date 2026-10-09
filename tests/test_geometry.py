from nasim.geometry import Position, distance


def test_position_defaults_z_to_zero():
    p = Position(1.0, 2.0)
    assert p.z == 0.0


def test_distance_2d():
    a = Position(0.0, 0.0)
    b = Position(3.0, 4.0)
    assert distance(a, b) == 5.0


def test_distance_includes_z_for_future_3d_use():
    a = Position(0.0, 0.0, 0.0)
    b = Position(0.0, 0.0, 5.0)
    assert distance(a, b) == 5.0
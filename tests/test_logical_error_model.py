import pytest

from nasim.logical_error_model import P_PH_REF, d_exp, logical_error_rate, logical_fidelity


def test_d_exponent_matches_ceil_d_plus_1_over_2():
    assert d_exp(3) == 2
    assert d_exp(5) == 3
    assert d_exp(7) == 4
    assert d_exp(9) == 5


def test_logical_error_rate_matches_literature_reference_at_d7_regardless_of_p_ph():
    # FT-Weave's own __post_init__ special-cases d == D_REF: the literal
    # reference table value is returned unchanged, the threshold formula
    # is never consulted at the reference distance - reproducing this
    # exactly (not "smoothing" it away) is what matches their published
    # numbers bit-for-bit.
    for op, ref in (("I", 2e-7), ("H", 4e-7), ("S", 1.46e-6)):
        assert logical_error_rate(op, p_ph=P_PH_REF, d=7) == ref
        assert logical_error_rate(op, p_ph=1e-4, d=7) == ref
        assert logical_error_rate(op, p_ph=5e-3, d=7) == ref


def test_logical_error_rate_scales_all_gate_types_by_the_same_ratio_away_from_d7():
    # Every gate type is scaled by the SAME ratio the (literature-fit)
    # CNOT formula predicts relative to the LITERAL table value - not
    # independently extrapolated per gate type.
    p_ph = 2e-4
    d = 5
    cnot_ratio = logical_error_rate("CNOT", p_ph, d) / logical_error_rate("CNOT", p_ph, 9)
    for op in ("I", "H", "S"):
        op_ratio = logical_error_rate(op, p_ph, d) / logical_error_rate(op, p_ph, 9)
        assert op_ratio == pytest.approx(cnot_ratio)


def test_logical_error_rate_decreases_with_distance_below_threshold_away_from_d7():
    p_ph = 1e-4  # well below P_TH = 8.5e-3
    for op in ("I", "H", "S", "CNOT"):
        rates = [logical_error_rate(op, p_ph, d) for d in (3, 5, 9, 11)]
        assert rates == sorted(rates, reverse=True)


def test_logical_error_rate_increases_with_distance_above_threshold_away_from_d7():
    p_ph = 2e-2  # above P_TH = 8.5e-3 - larger d should make it WORSE
    rates = [logical_error_rate("CNOT", p_ph, d) for d in (3, 5, 9, 11)]
    assert rates == sorted(rates)


def test_reproduces_ft_weave_csv_derived_rates_at_d_7_9_13():
    # Bit-for-bit against FT-Weave's own checked-in
    # output/evaluation/fidelity/t_cultivation_fidelity_results.csv,
    # inverting fidelity_X = f_X^n_X to recover per-gate base rates.
    assert logical_error_rate("CNOT", P_PH_REF, 7) == pytest.approx(3.0e-6)
    assert logical_error_rate("H", P_PH_REF, 7) == pytest.approx(4e-7)
    assert logical_error_rate("CNOT", P_PH_REF, 9) == pytest.approx(2.705193e-07, rel=1e-5)
    assert logical_error_rate("H", P_PH_REF, 9) == pytest.approx(3.606924e-08, rel=1e-5)
    assert logical_error_rate("CNOT", P_PH_REF, 13) == pytest.approx(1.995290e-09, rel=1e-5)
    assert logical_error_rate("H", P_PH_REF, 13) == pytest.approx(2.660387e-10, rel=1e-5)


def test_logical_fidelity_is_one_minus_error_rate():
    p_ph, d = 3e-4, 5
    for op in ("I", "H", "S", "CNOT"):
        assert logical_fidelity(op, p_ph, d) == pytest.approx(1.0 - logical_error_rate(op, p_ph, d))


def test_unknown_operation_raises():
    with pytest.raises(KeyError):
        logical_error_rate("T", p_ph=1e-4, d=7)

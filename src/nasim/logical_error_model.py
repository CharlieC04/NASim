from __future__ import annotations

import math

# const based on FT-WEAVE
# p_l(d) = A_CNOT * (p_ph / P_TH) ** ceil((d+1)/2)
P_TH = 8.5e-3
A_CNOT = 0.0579
D_REF = 7
P_PH_REF = 7.3e-4

# log err for d=7 from FT-WEAVE
_REF_RATES = {
    "I": 2e-7,
    "H": 4e-7,
    "S": 1.46e-6,
    "CNOT": 1.8e-6 + 2 * 4e-7 + 2 * 2e-7
}

def d_exp(d: int) -> int:
    return math.ceil((d+1)/2)

def logical_error_rate(op: str, p_ph: float, d: int) -> float:

    """
    LER of a single operation
    """

    ref = _REF_RATES[op]
    if d == D_REF: return ref

    new_cnot = A_CNOT * (p_ph / P_TH) ** d_exp(d)
    ratio = new_cnot / _REF_RATES["CNOT"]
    return ref * ratio

def logical_fidelity(op: str, p_ph: float, d: int) -> float:
    return 1.0 - logical_error_rate(op, p_ph, d)
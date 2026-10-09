# Idle-Qubit Management

Based on ZAP (Huang et al., IEEE TQE 2026), Eq. 12-15.

## The decision

A patch sitting in the entanglement zone between two of its own 2-qubit
gates has two options: **stay** (exposed to crosstalk from every merge that
happens around it in the meantime) or **return** to storage (costs a
round-trip transfer and movement time, but is crosstalk-free while parked).
`_idle_qubit_management` in `schedule.py` makes this decision per-qubit,
per-stage, by comparing expected fidelity loss under each option.

## The cost formulas

**Cost of staying** - accumulated crosstalk exposure over the $k$ stages
until the qubit's next 2-qubit use:
$$
\text{cost}_{\text{stay}} = k \cdot \big(-\ln f_{carrier}\big)
$$
If there's no future 2-qubit use at all ($k$ undefined), this is treated as
infinite.

**Cost of leaving** - a round trip costs
$n_{tr}$ transfers; if the qubit *will* be needed again
($k$ known), $n_{tr} = 4$ (out, settle, return, re-enter); if this was its
last use, $n_{tr} = 2$ (just leave):
$$
\text{cost}_{\text{leave}} = n_{tr} \cdot \big(-\ln f_{handover}\big) + \frac{n_{tr}}{2} \cdot \frac{t_{move}}{T_2}
$$
where $t_{move}$ is the single-trip movement duration (min-jerk model, see
[Movement & Routing](movement.md)) and $T_2$ the coherence time.

**Decision:** return to storage iff $\text{cost}_{\text{stay}} >
\text{cost}_{\text{leave}}$.
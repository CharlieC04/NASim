# Getting Started

## Install

```bash
git clone https://github.com/CharlieC04/NASim.git
cd NASim
pip install -e ".[dev,viz]"
```

`dev` pulls in `pytest` for the test suite; `viz` pulls in `matplotlib` for
the visualizer. Both are optional - the core simulator only depends on
`numpy` and `pyyaml`.

Verify the install:

```bash
pytest -q
```

## A minimal end-to-end run

This builds a 4-qubit circuit as a flat gate list, simulates
it on a non-zoned device, and prints the resulting timing and fidelity.
Note that the fidelity result is a WIP.

```python
from nasim.circuit import Circuit, one_qubit, two_qubit
from nasim.device import Device
from nasim.model import Model
from nasim.placement import patch_separation_um, piqasso_placement
from nasim.schedule import circuit_fidelity, simulate_circuit

model = Model.from_yaml("src/nasim/config/presets/2d.yaml")

gates = [
    one_qubit(0), one_qubit(1), one_qubit(2), one_qubit(3),
    two_qubit(0, 1), two_qubit(2, 3),
    two_qubit(1, 2),
]
circuit = Circuit.from_gates(4, gates)  # ASAP-separate scheduling

device = Device.uniform(
    model=model, rows=20, cols=20,
    pitch_um=patch_separation_um(model, distance=3),
)
placement = piqasso_placement(circuit, model, device, distance=3, seed=0)

schedule, final_placement = simulate_circuit(circuit, placement, model)

print(f"Ng1={schedule.Ng1} Ng2={schedule.Ng2} Nh={schedule.Nh} clock={schedule.clock_us:.2f}us")
print(f"fidelity: {circuit_fidelity(schedule, d=3):.4g}")
```

For a **zoned** device (separate storage/entanglement regions), swap
`piqasso_placement` for `zap_placement` and pass `device=device` into
`simulate_circuit` - see [Zoned Architecture](concepts/zoned-architecture.md)
and [examples/zone_demo.py](https://github.com/CharlieC04/NASim/blob/main/examples/zone_demo.py).

## Next steps

- Browse the [Examples](examples.md) gallery - every script in `examples/`
  demonstrates one concept in isolation.
- Read [Overview & Pipeline](concepts/overview.md) for how the pieces above
  fit together.

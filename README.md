# NASim

[![tests](https://github.com/CharlieC04/NASim/actions/workflows/test.yml/badge.svg)](https://github.com/CharlieC04/NASim/actions/workflows/test.yml)
[![docs](https://github.com/CharlieC04/NASim/actions/workflows/docs.yml/badge.svg)](https://charliec04.github.io/NASim/)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)

A simulator for compiling quantum circuits onto neutral-atom fault-tolerant
hardware. NASim takes a logical circuit, schedules it, places it onto a
physical device, routes the atom movement needed to execute it, and reports
the resulting timing and fidelity.

The end goal is a simulator for developing **novel 3D neutral-atom
compilation methods** - true 3D, with interacting layers. The current phase
deliberately targets **2D zoned architectures** first, validating
scheduling, placement, routing and idle-qubit management against the
published literature before extending to a stack of interacting layers.

**[Full documentation →](https://charliec04.github.io/NASim/)**

## What it does

- **ASAP-separate scheduling** - turns a flat gate list into a
  hardware-executable, pure-stage schedule.
- **Two placement algorithms** - a force-directed layout (Piqasso) and a
  deterministic, routing-aware one (ZAP), for any device or zoned devices
  respectively.
- **AOD-aware movement routing**, including a check for atoms that would be
  accidentally swept up mid-move.
- **Lattice surgery** - logical operations as rotated-surface-code
  merge/split, validated against a published reference result.
- **Zoned architecture** - separate storage/entanglement regions with
  crosstalk accounting and a cost-based idle-qubit stay/return policy.
  
## Install

```bash
git clone https://github.com/CharlieC04/NASim.git
cd NASim
pip install -e ".[dev,viz]"
```

`dev` pulls in `pytest` for the test suite; `viz` pulls in `matplotlib` for
the visualizer. The core simulator only depends on `numpy` and `pyyaml`.

Verify the install:

```bash
pytest -q
```

## Quickstart

```python
from nasim.circuit import Circuit, one_qubit, two_qubit
from nasim.device import Device
from nasim.model import Model
from nasim.placement import patch_separation_um, piqasso_placement
from nasim.schedule import circuit_fidelity, simulate_circuit

model = Model.from_yaml("src/nasim/config/presets/2d.yaml")

circuit = Circuit.from_gates(4, [
    one_qubit(0), one_qubit(1), one_qubit(2), one_qubit(3),
    two_qubit(0, 1), two_qubit(2, 3),
    two_qubit(1, 2),
])

device = Device.uniform(
    model=model, rows=20, cols=20,
    pitch_um=patch_separation_um(model, distance=3),
)
placement = piqasso_placement(circuit, model, device, distance=3, seed=0)

schedule, final_placement = simulate_circuit(circuit, placement, model)
print(f"Nh={schedule.Nh} clock={schedule.clock_us:.2f}us")
print(f"fidelity: {circuit_fidelity(schedule, d=3):.4g}")
```

More in [`examples/`](examples/), see the [Examples gallery](https://charliec04.github.io/NASim/examples/)
for a guided walkthrough.

## License

[GNU GPLv3](LICENSE)

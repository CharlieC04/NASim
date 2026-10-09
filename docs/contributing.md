# Contributing

## Setup

```bash
git clone https://github.com/CharlieC04/NASim.git
cd NASim
pip install -e ".[dev,viz]"
pytest -q
```

## Running things

```bash
pytest -q                          # full test suite
pytest tests/test_schedule.py -q   # one module
python examples/zone_demo.py       # run an example end-to-end
```

## Where things live

| If you're changing... | Look at |
|---|---|
| How circuits are scheduled | `circuit.py`, [Scheduling](concepts/scheduling.md) |
| How qubits get an initial position | `placement.py`, [Placement](concepts/placement.md) |
| Device/zone geometry | `device.py`, [Zoned Architecture](concepts/zoned-architecture.md) |
| Atom transport legality | `move.py`, [Movement & Routing](concepts/movement.md) |
| The core simulation loop, idle-qubit decisions, fidelity | `schedule.py` |
| Surface-code patch layout | `surface_code.py` |
| Merge/split mechanics | `lattice_surgery.py` |
| Visualization | `viz/` |

## Adding a new algorithm or paper-derived formula

1. Add it to the relevant Concepts page with the actual equation and a
   citation (arXiv ID or DOI).
2. Add it to the [Sources](sources.md) table.
3. If it changes observable behaviour, add or update a test.

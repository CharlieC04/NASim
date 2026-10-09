# Examples

Every script below is runnable directly: `python examples/<name>.py`. Most
visualization-producing ones save frames/a GIF into a directory next to
where you run them.

| Script | Demonstrates |
|---|---|
| `asap_separate_demo.py` | [ASAP-separate scheduling](concepts/scheduling.md) on a deliberately irregular gate list, plus an end-to-end simulation. |
| `zone_demo.py` | A flat 6-qubit gate list, `Circuit.from_gates`, `zap_placement`, three simultaneous disjoint merges, then idle-qubit stay/return for the two qubits not needed again. |
| `qaoa_chain.py` | A QAOA/Ising brick-wall ansatz on a qubit chain. |
| `surgery_cnot.py` | Validates [lattice surgery](concepts/lattice-surgery.md) against Viszlai et al.'s published measurement-round and qubit-overhead claims. |

## Reading the visualizer output

`ScheduleViewer` (`viz/schedule.py`) collapses a `Schedule`'s raw operation
trace into display frames and can render them to PNGs
(`save_frames`), a GIF (`save_gif`), or an interactive window (`show`). Zone
boundaries, patch bounding boxes, and data/ancilla atom roles are all drawn
automatically when a zoned `Device` is passed in.

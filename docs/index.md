# NASim

**NASim** is a simulator for compiling quantum circuits onto neutral-atom
fault-tolerant hardware. It takes a logical circuit, schedules it, places it
onto a physical device, routes the atoms, and
reports the resulting timing and fidelity.

The project's goal is a simulator for developing **novel 3D neutral-atom
compilation methods**. The current phase targets **2D zoned architectures**
to establish a baseline.

## What it does today

- Compiles a flat list of gates into a hardware-executable schedule via
  **ASAP-separate scheduling** (see [Scheduling](concepts/scheduling.md)).
- Places logical qubits onto a device either with a physics-inspired
  force-directed layout or a deterministic, routing-aware algorithm (see
  [Placement](concepts/placement.md)).
- Routes atom movement with an AOD-aware conflict model (see
  [Movement & Routing](concepts/movement.md)).
- Executes logical operations as rotated-surface-code lattice surgery (see
  [Lattice Surgery](concepts/lattice-surgery.md)).
- Models a **zoned architecture** with crosstalk accounting and a 
  cost-based idle-qubit stay/return policy
  (see [Zoned Architecture](concepts/zoned-architecture.md) and
  [Idle-Qubit Management](concepts/idle-management.md)).
- Estimates circuit fidelity with a hybrid model: physical-gate terms for
  everything outside the code, and surface-code threshold scaling for the
  syndrome-extraction rounds themselves (work in progress).

# Surface Codes & Lattice Surgery

## Patches

A logical qubit is a **rotated surface-code patch** (`Patch.rotated(distance)`
in `surface_code.py`). For an odd code distance $d$, a patch lays out
$d \times d$ data atoms on a grid with spacing 2 (so ancilla sites can sit at
the odd coordinates between them), plus interior X/Z ancilla atoms in a
checkerboard pattern, plus boundary ancillas along each edge.

Every atom is a `PatchAtom(role, local)`, where `role` is `DATA`,
`ANCILLA_X`, or `ANCILLA_Z`, and `local` is an integer `(x, y)` on that
spacing-2 grid. `Patch.embed(anchor, unit_um)` converts local coordinates to
absolute device positions given a physical anchor and the unit distance
between neighbouring atoms (`model.gate_pair_dist_um`).

Each of a patch's four edges carries a fixed ancilla type
(`boundary_role`): top/bottom edges are Z-type, left/right are X-type - this
is what the surgery code uses to check two patches can actually be merged.

## Merging two patches (`lattice_surgery.py`)

Two patches can be merged only if they're physically adjacent with matching
extents (`facing_edges`) **and** their facing edges carry the same ancilla
type (`check_alignment`). If both checks pass, `merge_patches` builds
the merged patch as a single larger rectangle (`2d x d` or `d x 2d`
depending on merge orientation) anchored at whichever of the two original
patches sits at the merged patch's own local origin.

## Executing a merge (`schedule.py::record_merge` / `record_split`)

1. Compute which atoms are new (the bridge region between the two patches)
   and which are removed (duplicated boundary atoms absorbed into the
   merge) by diffing embedded positions before/after.
2. Move the bridge atoms into place (`merge_handover`), costed by handover
   time, same as any other movement.
3. Run syndrome extraction for `distance` rounds. Each round is 4 sequential
   CZ sub-steps (interior ancillas fire every round, boundary ancillas only
   fire in the first 2) followed by one mid-circuit measurement.
4. `record_split` reverses the process back to two independent patches.

## Validated against the literature

[`examples/surgery_cnot.py`](https://github.com/CharlieC04/NASim/blob/main/examples/surgery_cnot.py)
reproduces a CNOT via lattice surgery (merge, measure, split) and checks two
claims from Viszlai et al., *"An Architecture for Improved Surface Code
Connectivity in Neutral Atoms"* (arXiv:2309.13507), Section 5.1.3 / Fig. 5:

- **$2d$ total measurement rounds** for a full merge+split CNOT at distance $d$.
- **1.5x-2x physical qubit overhead** versus a transversal CNOT (the lattice
  surgery patch needs $3 \times (\text{atoms per standalone patch})$ versus
  $2 \times$ for two independent patches).

Both checks pass for $d = 3, 5, 7$.

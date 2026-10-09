# Movement & Routing

Based on the AOD (acousto-optic deflector) transport model shared by ZAC and
ZAP, plus the unintended-pickup constraint from FT-Weave (Lin, Kornjača,
Zhao, Wang, Cong, 2026, Section IV-D).

## How AOD transport works

An AOD moves atoms by activating a set of row-tones and column-tones
simultaneously; every atom sitting at the **Cartesian product** of those
tones gets picked up together, and atoms sharing a tone move together along
that axis. This gives three physical constraints on which moves can share
one pulse ("frame"):

1. **Trajectories must not cross.**
2. **Atoms sharing a source row/column must be deposited into the same
   destination row/column**.
3. **The activated tones must not sweep up an atom that wasn't supposed to
   move**.

## Move representation & duration (`move.py`)

```python
@dataclass(frozen=True)
class Move:
    qubit: int
    source: Position
    target: Position
    clearance_um: float = 0.0
```

Duration follows Piqasso's min-jerk model:
$$ t_{mv} = \frac{15}{8} \cdot \frac{d}{v_{xy}} $$
where $d$ is the straight-line distance and $v_{xy}$ the device's movement
speed (`model.v_xy_um_per_us`).

## Pairwise compatibility - constraints 1 & 2 (`compatible_2d`)

`conflict_reason(a, b)` checks each axis independently for a shared-source
(diverging targets), shared-target (diverging sources), or crossing
conflict. `compatible_2d` additionally enforces a clearance margin between
patch footprints. Incompatible moves can't share a frame.

## Frame legalisation (`legalise_frames`)

Given a list of moves for one stage, `legalise_frames` repeatedly extracts
one maximal compatible set as a frame until every move is scheduled. A move that's blocked by
exactly one *shared-source* conflict can be **parked** - nudged a short
clearance-distance sideways first, so its remainder becomes compatible and
it doesn't have to wait a full extra frame.

## Unintended-pickup safety - constraint 3 (`_aod_violations` / `_resolve_aod_violations`)

The pairwise checks above only ever compared moves *against each other* -
they had no idea whether some other, entirely stationary patch happened to
sit at a Cartesian-product intersection of the chosen frame's source tones.

> Move A: `(0, 0) -> (0, 100)`, Move B: `(50, 200) -> (50, 300)`. Source
> tones are `x ∈ {0, 50}`, `y ∈ {0, 200}` - the spurious combination
> `(0, 200)` is neither move's own source. If a third, stationary patch
> sits there, firing A and B together would illegally sweep it up.

`legalise_frames` takes an optional `stationary` list (every patch
*not* moving at all this stage); `_build_frame_with_parking` checks the
chosen frame against it and, if a violation exists, greedily removes
whichever single move clears the most violations, deferring it to a later
frame, until none remain.

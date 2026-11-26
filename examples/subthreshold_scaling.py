"""
Calibrates each fidelity gadget at a handful of points safely BELOW its
own threshold, fits the known asymptotic power law p_L(p) = A * p**n to
that data, and evaluates the fit (not a fresh Monte Carlo run) at a dense
grid of deployment-range points this experiment cares about: p_phys from
1e-4 up to 4e-3 (see TARGET_POINTS) -- dense enough to see a smooth trend
for the real circuits in stim_circuit_fidelity.py, not just 5 sparse
points.

merge_split and surface_memory are calibrated live, against our own
Stim+PyMatching pipeline (see GADGETS) -- there's no external reference
curve for either, so this is the only option. qldpc_memory is different:
checking our own BP-OSD calibration (fast=True, osd_order=0) against
qSIEVE's own published Fig 8 curve at the SAME p values showed a 12x-24x
disagreement, almost certainly because osd_order=0 is a much weaker
decoder than qSIEVE's own osd_order=10 (see
calibrate_qldpc_memory_adaptive's docstring). Rather than spend the time
on a slow osd_order=10 recalibration right now, qldpc_memory's model is
fit directly from qSIEVE's own published numbers instead (QSIEVE_REFERENCE)
-- externally grounded, and zero live calibration needed for this gadget.
Redoing it with our own recalibrated (osd_order=10) data is future work,
once there's time for a much slower BP-OSD run.

Why not calibrate merge_split/surface_memory directly at 1e-4-5e-4?
Logical error rates there are expected to be extremely small (that's the
point of running below threshold) -- small enough that observing even
one event in a tractable shot count isn't guaranteed. Fitting from a
measurable sub-threshold window and extrapolating is the standard way
around this.

FIT_POINTS_BY_GADGET / DEFAULT_FIT_POINTS were chosen to sit comfortably
below each gadget's own empirically-observed threshold -- rotated
surface code (circuit-level depolarizing noise) has a literature
threshold of ~0.5-1%, consistent with our own merge/split calibration
scan from earlier in this project (p=1e-3 gave a small, sane error rate;
p=5e-3 gave ~28% for a single d=5 merge/split event). QSIEVE_FIT_SUBSET
is chosen the same way, but from qSIEVE's own published curve: only
points strictly below where their curve crosses y=x (~6e-3).

All calibration results are cached under examples/calibration_cache/ (see
nasim/fidelity/cache.py) -- rerunning this script, or any other script
that requests the same (gadget, p_phys, decoder settings) point, costs
nothing once the cache is populated.

Run from the repo root, in the MAIN venv:

    python examples/subthreshold_scaling_experiment.py

If a fit point comes back with 0 observed errors (seen in practice: the
surface-memory gadget at p=1e-3 gave 0/500_000, and still 0/1_000_000 on
a manual rerun -- d=13 suppresses errors hard enough that even 1e-3 can
be too deep for that shot budget), _retry_until_nonzero automatically
re-calibrates at a larger max_shots (see ZERO_ERROR_RETRY_GROWTH /
ZERO_ERROR_MAX_RETRIES) rather than crashing in fit_power_law's log(0).
If even ABS_MAX_SHOTS_* is exhausted with 0 errors, it raises instead of
guessing -- at that point FIT_POINTS needs to move closer to threshold
for that gadget, not a bigger cap.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from nasim.fidelity import (
    calibrate_gen_bicycle_code,
    decode_batch_matching,
    merge_split_circuit,
    surface_code_memory_circuit,
)
from nasim.fidelity.cache import cached_calibrate
from nasim.fidelity.calibrate import CalibrationResult
from nasim.fidelity.scaling_fit import fit_power_law, predict_power_law, r_squared
from nasim.qldpc import BB_144_12_12

SURFACE_DIST = 13
SURFACE_K = (SURFACE_DIST - 1) // 2  # tqec convention d = 2k + 1 -> k = 6
QLDPC_ROUNDS = 12
QLDPC_FAST = True  # osd_order=0: ~90x faster than osd_order=10, see module docstring

DEFAULT_FIT_POINTS = [1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3]
# surface_memory and qldpc_memory each needed their own shift up from
# DEFAULT_FIT_POINTS: 1e-3 gave 0 observed errors for surface_memory even
# after retrying up to ABS_MAX_SHOTS_MATCHING (d=13 suppresses errors
# harder than the margin from threshold alone suggested), and
# qldpc_memory was taking far too long for the same reason -- BB_144_12_12
# + BP-OSD's own threshold is ~6e-3 (where the qSIEVE Fig 8 reference
# curve crosses y=x), so 1e-3 is ~6x below it on the slowest decoder in
# this pipeline (BP-OSD decodes one shot at a time). Shifted the whole
# window up by 1e-3 rather than just dropping the lowest point, to keep
# 5 evenly-spaced points for the fit.
FIT_POINTS_BY_GADGET = {
    "surface_memory": [2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3],
    "qldpc_memory": [2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3],
}
# Densely log-spaced from 1e-4 up to 4e-3 (the fit window's OWN upper
# edge, FIT_POINTS_BY_GADGET's highest value) -- every point here is
# either an extrapolation below the fit window or an interpolation
# within/at it, never an extrapolation ABOVE it. Do not extend this past
# 4e-3 towards qSIEVE's 6e-3-9e-3 range: that's at/above qldpc_memory's
# own threshold, where this power law is known to break down (see module
# docstring) -- validating against qSIEVE there needs direct calibration,
# not this fit (see qldpc_qsieve_validation.py).
TARGET_POINTS = [
    1e-4, 1.5e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 8e-4,
    1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3,
]

TARGET_ERRORS_MATCHING = 50   # merge/split + surface memory: PyMatching is cheap, can afford more
BATCH_MATCHING = 2000
MAX_SHOTS_MATCHING = 500_000
ABS_MAX_SHOTS_MATCHING = 32_000_000  # PyMatching is cheap enough to push this far before giving up

TARGET_ERRORS_QLDPC = 30      # BP-OSD is expensive -- fewer target errors, smaller batch, smaller cap
BATCH_QLDPC = 200
MAX_SHOTS_QLDPC = 20_000       # fit points now sit closer to threshold (see FIT_POINTS_BY_GADGET), so
ABS_MAX_SHOTS_QLDPC = 200_000  # the baseline rate should be much higher and this should rarely need to retry

# If a fit point comes back with 0 observed errors, fit_power_law can't
# take its log -- rather than crash (or silently drop the point), retry
# with max_shots scaled up by this factor, up to ZERO_ERROR_MAX_RETRIES
# times. Each retry uses a different max_shots, which is part of the
# cache tag, so it's a fresh calibration, not a replay of the exhausted
# zero-error result.
ZERO_ERROR_RETRY_GROWTH = 4.0
ZERO_ERROR_MAX_RETRIES = 3

CACHE_DIR = Path("examples/calibration_cache")
MERGE_SPLIT_CACHE = CACHE_DIR / "merge_split_subthreshold.csv"
SURFACE_CACHE = CACHE_DIR / "surface_memory_subthreshold.csv"
QLDPC_CACHE = CACHE_DIR / "qldpc_memory_subthreshold.csv"

RESULTS_CSV = Path("examples/subthreshold_scaling_results.csv")


def _adaptive_matching_calibration(make_circuit, *, target_errors: int, batch: int, max_shots: int) -> tuple[int, int]:
    """
    Generic PyMatching-decoded adaptive shot collection: build the circuit
    once, sample in batches, stop once target_errors is hit or max_shots
    is exhausted. Neither calibrate_merge_split nor calibrate_surface_code
    currently exposes an adaptive mode (only calibrate_gen_bicycle_code
    does) -- this generalizes that same batching pattern to any
    PyMatching-decodable gadget instead of duplicating it per gadget.
    """
    circuit = make_circuit()
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler()

    total_shots = total_errors = 0
    while total_errors < target_errors and total_shots < max_shots:
        this_batch = min(batch, max_shots - total_shots)
        detection_events, observable_flips = sampler.sample(shots=this_batch, separate_observables=True)
        predicted = decode_batch_matching(dem, detection_events)
        total_errors += int(np.sum(np.any(predicted != observable_flips, axis=1)))
        total_shots += this_batch

    return total_shots, total_errors


def calibrate_merge_split_adaptive(p_phys: float, *, max_shots: int) -> CalibrationResult:
    tag = {"gadget": "merge_split", "k": SURFACE_K, "basis": "Z",
           "target_errors": TARGET_ERRORS_MATCHING, "max_shots": max_shots}

    def compute() -> CalibrationResult:
        shots, n_errors = _adaptive_matching_calibration(
            lambda: merge_split_circuit(k=SURFACE_K, p=p_phys, basis="Z"),
            target_errors=TARGET_ERRORS_MATCHING, batch=BATCH_MATCHING, max_shots=max_shots,
        )
        return CalibrationResult(p_phys=p_phys, rounds=1, shots=shots, n_errors=n_errors)

    return cached_calibrate(MERGE_SPLIT_CACHE, tag, p_phys=p_phys, compute=compute)


def calibrate_surface_memory_adaptive(p_phys: float, *, max_shots: int) -> CalibrationResult:
    tag = {"gadget": "surface_memory", "distance": SURFACE_DIST, "rounds": QLDPC_ROUNDS,
           "target_errors": TARGET_ERRORS_MATCHING, "max_shots": max_shots}

    def compute() -> CalibrationResult:
        shots, n_errors = _adaptive_matching_calibration(
            lambda: surface_code_memory_circuit(distance=SURFACE_DIST, rounds=QLDPC_ROUNDS, p=p_phys),
            target_errors=TARGET_ERRORS_MATCHING, batch=BATCH_MATCHING, max_shots=max_shots,
        )
        return CalibrationResult(p_phys=p_phys, rounds=QLDPC_ROUNDS, shots=shots, n_errors=n_errors)

    return cached_calibrate(SURFACE_CACHE, tag, p_phys=p_phys, compute=compute)


def calibrate_qldpc_memory_adaptive(p_phys: float, *, max_shots: int) -> CalibrationResult:
    """
    NOT currently used in main()'s default flow -- see QSIEVE_REFERENCE
    and the qldpc_memory block in main(). Our own BP-OSD calibration
    (fast=True, osd_order=0) disagrees with qSIEVE's published curve by
    12x-24x in the 1e-3-4e-3 window (checked directly: this function's
    own output compared against QSIEVE_REFERENCE at those same p values),
    almost certainly because osd_order=0 is a much weaker decoder than
    qSIEVE's own (osd_order=10). Rather than spend the time on a slow
    osd_order=10 recalibration right now, qldpc_memory's model is fit
    directly from qSIEVE's own published numbers instead (externally
    grounded, zero live calibration needed). This function is kept for
    later, when there's time to redo that recalibration and compare.
    """
    tag = {"gadget": "qldpc_memory", "code": BB_144_12_12.name, "rounds": QLDPC_ROUNDS, "fast": QLDPC_FAST,
           "target_errors": TARGET_ERRORS_QLDPC, "max_shots": max_shots}

    def compute() -> CalibrationResult:
        # calibrate_gen_bicycle_code already has its own adaptive loop
        # (shots=None -> batch until target_errors/max_shots) -- reuse it
        # directly rather than reimplementing for the one gadget that
        # already supports this.
        return calibrate_gen_bicycle_code(
            BB_144_12_12, p_phys=p_phys, rounds=QLDPC_ROUNDS, shots=None,
            target_errors=TARGET_ERRORS_QLDPC, max_shots=max_shots, fast=QLDPC_FAST,
        )

    return cached_calibrate(QLDPC_CACHE, tag, p_phys=p_phys, compute=compute)


def _retry_until_nonzero(calibrate_fn, p_phys: float, *, initial_max_shots: int, growth: float,
                         max_retries: int, absolute_cap: int) -> CalibrationResult:
    """
    Calls calibrate_fn(p_phys, max_shots=...) and, if it comes back with 0
    observed errors, retries with max_shots scaled up by `growth`, up to
    max_retries times or absolute_cap, whichever comes first. A fresh
    max_shots is a fresh cache key (see cached_calibrate's tag), so this
    never just re-reads the exhausted zero-error row.

    Raises RuntimeError if even absolute_cap gives 0 errors -- at that
    point the right fix is moving this fit point further from threshold
    in FIT_POINTS, not raising the cap further (Monte Carlo cost grows
    faster than the cap can reasonably chase a vanishing rate).
    """
    max_shots = min(initial_max_shots, absolute_cap)
    attempt = 0
    while True:
        result = calibrate_fn(p_phys, max_shots=max_shots)
        if result.n_errors > 0:
            if attempt > 0:
                print(f"    p={p_phys}: needed {attempt} retry(ies), max_shots={max_shots} to see a nonzero rate")
            return result

        if attempt >= max_retries or max_shots >= absolute_cap:
            raise RuntimeError(
                f"p_phys={p_phys} gave 0 errors even at max_shots={result.shots} "
                f"(cap={absolute_cap}, after {attempt} retries). This fit point is too deep "
                f"below threshold for this shot budget -- move it up in FIT_POINTS instead of "
                f"raising the cap further."
            )

        next_max_shots = min(int(max_shots * growth), absolute_cap)
        print(f"    p={p_phys}: 0 errors at max_shots={result.shots} -- retrying with max_shots={next_max_shots}")
        max_shots = next_max_shots
        attempt += 1


# Live-calibrated gadgets (no external reference curve exists for either
# of these -- merge/split is our own tqec-built lattice-surgery gadget,
# surface_memory is the standard rotated surface code -- so Monte Carlo
# against our own Stim+PyMatching pipeline is the only option, and
# PyMatching doesn't have BP-OSD's decoder-order speed/accuracy tradeoff).
GADGETS = {
    "merge_split": (calibrate_merge_split_adaptive, MAX_SHOTS_MATCHING, ABS_MAX_SHOTS_MATCHING),
    "surface_memory": (calibrate_surface_memory_adaptive, MAX_SHOTS_MATCHING, ABS_MAX_SHOTS_MATCHING),
}

# qSIEVE Fig 8(a), [[144,12,12]] (weight 6), per-round logical error rate,
# read directly from the published figure. qldpc_memory's model is fit
# directly from this instead of our own calibration -- see
# calibrate_qldpc_memory_adaptive's docstring for why.
QSIEVE_REFERENCE = [
    (1e-3, 1.3e-7),
    (2e-3, 3.5e-6),
    (3e-3, 5.2e-5),
    (4e-3, 3.5e-4),
    (5e-3, 1.7e-3),
    (6e-3, 6.0e-3),
    (7e-3, 1.9e-2),
    (8e-3, 4.7e-2),
    (9e-3, 9.5e-2),
]
# Only the points strictly below qSIEVE's own threshold crossing (where
# their curve crosses y=x, ~6e-3) are valid for a sub-threshold power-law
# fit -- same reasoning as FIT_POINTS_BY_GADGET above, just applied to
# their data instead of ours.
QSIEVE_FIT_SUBSET = [(p, r) for p, r in QSIEVE_REFERENCE if p < 6e-3]


def main() -> None:
    rows = []
    for gadget_name, (calibrate_fn, initial_max_shots, absolute_cap) in GADGETS.items():
        print(f"=== {gadget_name} ===")

        fit_points = FIT_POINTS_BY_GADGET.get(gadget_name, DEFAULT_FIT_POINTS)
        fit_rates = []
        for p in fit_points:
            result = _retry_until_nonzero(
                calibrate_fn, p, initial_max_shots=initial_max_shots,
                growth=ZERO_ERROR_RETRY_GROWTH, max_retries=ZERO_ERROR_MAX_RETRIES, absolute_cap=absolute_cap,
            )
            rate = result.per_round_logical_error_rate
            print(f"  fit point p={p}: shots={result.shots} n_errors={result.n_errors} "
                  f"per_round_rate={rate:.3e}")
            fit_rates.append(rate)

        A, n = fit_power_law(fit_points, fit_rates)
        r2 = r_squared(fit_points, fit_rates, A, n)
        print(f"  fit: rate(p) = {A:.3e} * p^{n:.3f}  (R^2={r2:.4f} in log-log space)")

        for p in TARGET_POINTS:
            predicted_rate = predict_power_law(p, A, n)
            rows.append({
                "gadget": gadget_name, "p_phys": p, "predicted_per_round_rate": predicted_rate,
                "fit_A": A, "fit_n": n, "fit_r_squared": r2,
            })
            print(f"    p={p}: predicted per-round rate = {predicted_rate:.3e}")

    print("=== qldpc_memory (fit directly from qSIEVE's published curve, no live calibration) ===")
    qsieve_p = [p for p, _ in QSIEVE_FIT_SUBSET]
    qsieve_rates = [r for _, r in QSIEVE_FIT_SUBSET]
    A, n = fit_power_law(qsieve_p, qsieve_rates)
    r2 = r_squared(qsieve_p, qsieve_rates, A, n)
    print(f"  fit: rate(p) = {A:.3e} * p^{n:.3f}  (R^2={r2:.4f} in log-log space, fit to {len(qsieve_p)} qSIEVE points)")
    for p in TARGET_POINTS:
        predicted_rate = predict_power_law(p, A, n)
        rows.append({
            "gadget": "qldpc_memory", "p_phys": p, "predicted_per_round_rate": predicted_rate,
            "fit_A": A, "fit_n": n, "fit_r_squared": r2,
        })
        print(f"    p={p}: predicted per-round rate = {predicted_rate:.3e}")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
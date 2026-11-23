"""
Calibrate BB_144_12_12's per-round logical error rate via BP+OSD and
compare against qSIEVE's own published curve (Sec 6, Fig 8, Table 1).

Matches qSIEVE's own methodology (Sec 6.2/6.4/8.2) directly:
  - min-sum BP, OSD order 10 (the default here; --fast switches to OSD-0
    for a much quicker but weaker-decoder sanity pass instead)
  - X and Z checks decoded as separate sub-problems, not jointly
  - adaptive shot collection: keep sampling until enough logical errors
    are seen, rather than a fixed shot count -- their own rule is "at
    least 1,000 errors, or 1e9 shots, every point got at least 100
    errors." The defaults here (--target-errors 20, --max-shots 2000) are
    far smaller, since our reference BP-OSD decoder is much slower
    per-shot than their production setup; raise both for a tighter
    estimate once you know how long a given (p, rounds) point takes on
    your machine. Pass --shots N instead for a fixed-budget run (skips
    adaptive collection entirely).

Run from the repo root:
    python examples/qldpc_fidelity_calibration.py
    python examples/qldpc_fidelity_calibration.py --fast
    python examples/qldpc_fidelity_calibration.py --target-errors 50 --max-shots 5000
    python examples/qldpc_fidelity_calibration.py --shots 200
"""
import argparse
import csv
import time
from pathlib import Path

from nasim.fidelity import calibrate_gen_bicycle_code
from nasim.qldpc import BB_144_12_12

RESULTS_CSV = Path("examples/qldpc_fidelity_results.csv")
P_PHYS_POINTS = [6e-3, 7e-3, 8e-3, 9e-3]
ROUNDS = 12  # = code distance, matching qSIEVE's "standard d parity check cycles"

# Measured on this machine at p=1e-3, rounds=12, osd_order=10 -- used only
# to print a rough estimate up front; actual per-point timing is reported
# as it runs, and adaptive mode's total time depends on how often errors
# actually occur, not just this per-shot figure.
PER_SHOT_ESTIMATE_FAST = 0.35
PER_SHOT_ESTIMATE_ACCURATE = 2.75


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="osd_order=0 (fast, weaker decoder) instead of osd_order=10 (qSIEVE's setting, slow)")
    parser.add_argument("--shots", type=int, default=None,
                        help="fixed shot count (skips adaptive collection)")
    parser.add_argument("--target-errors", type=int, default=20,
                        help="adaptive mode: stop once this many logical errors are seen (qSIEVE uses 1000)")
    parser.add_argument("--max-shots", type=int, default=2000,
                        help="adaptive mode: stop after this many shots regardless (qSIEVE uses 1e9)")
    args = parser.parse_args()

    per_shot_estimate = PER_SHOT_ESTIMATE_FAST if args.fast else PER_SHOT_ESTIMATE_ACCURATE
    if args.shots is not None:
        est_total = len(P_PHYS_POINTS) * args.shots * per_shot_estimate
        print(f"Estimated total runtime: ~{est_total:.0f}s (~{est_total/60:.1f} min) for "
              f"{len(P_PHYS_POINTS)} points x {args.shots} fixed shots, "
              f"{'fast/osd_order=0' if args.fast else 'osd_order=10 (qSIEVE match)'}\n")
    else:
        est_worst_case = len(P_PHYS_POINTS) * args.max_shots * per_shot_estimate
        print(f"Adaptive mode: up to {args.max_shots} shots/point or {args.target_errors} errors, "
              f"whichever comes first. Worst case (no errors ever found): "
              f"~{est_worst_case:.0f}s (~{est_worst_case/60:.1f} min) across {len(P_PHYS_POINTS)} points. "
              f"Typically much faster once points near/above threshold start producing errors quickly.\n")

    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "code", "p_phys", "rounds", "shots", "n_errors",
                "logical_error_rate", "per_round_logical_error_rate",
            ])

        for p in P_PHYS_POINTS:
            t0 = time.time()
            result = calibrate_gen_bicycle_code(
                BB_144_12_12, p_phys=p, rounds=ROUNDS, shots=args.shots,
                target_errors=args.target_errors, max_shots=args.max_shots, fast=args.fast,
            )
            elapsed = time.time() - t0
            print(f"p={p:.4f}: logical_error_rate={result.logical_error_rate:.4f}  "
                  f"per_round={result.per_round_logical_error_rate:.6f}  "
                  f"(n_errors={result.n_errors}/{result.shots}, {elapsed:.1f}s)")
            writer.writerow([
                BB_144_12_12.name, p, ROUNDS, result.shots, result.n_errors,
                result.logical_error_rate, result.per_round_logical_error_rate,
            ])
            f.flush()

    print(f"\nWrote results to {RESULTS_CSV}")
    print("Compare per_round_logical_error_rate against qSIEVE's Fig 8 curve for [[144,12,12]] "
          "(Sec 6.4). The circuit previously had a bug (no final check round reconstructed "
          "from the data measurement) giving an effective distance of 1 -- now fixed; expect "
          "results much closer to the published curve than earlier runs.")


if __name__ == "__main__":
    main()

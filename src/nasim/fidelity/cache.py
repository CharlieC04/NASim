from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from nasim.fidelity.calibrate import CalibrationResult


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _matches(row: dict[str, str], key: dict[str, object]) -> bool:
    return all(row.get(k) == str(v) for k, v in key.items())


def cached_calibrate(csv_path: Path, tag: dict[str, object], *, p_phys: float,
                     compute: Callable[[], CalibrationResult]) -> CalibrationResult:
    
    key = {**tag, "p_phys": p_phys}
    for row in _read_rows(csv_path):
        if _matches(row, key):
            return CalibrationResult(p_phys=p_phys, rounds=int(row["rounds"]),
                                      shots=int(row["shots"]), n_errors=int(row["n_errors"]))

    result = compute()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    fieldnames = list(tag.keys()) + ["p_phys", "rounds", "shots", "n_errors"]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({**{k: str(v) for k, v in tag.items()}, "p_phys": p_phys,
                         "rounds": result.rounds, "shots": result.shots, "n_errors": result.n_errors})
    return result
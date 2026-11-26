from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "ftweave_tcultivation"
RESULTS_CSV = FIXTURES_DIR / "t_cultivation_fidelity_results.csv"
PROFILING_CSV = FIXTURES_DIR / "t_cultivation_fidelity_profiling_results.csv"

_LAYOUT_RE = re.compile(r"\((\d+),\s*(\d+)\)")


def parse_qubit_layout(layout: str) -> tuple[int, int]:

    m = _LAYOUT_RE.match(layout)
    if not m:
        raise ValueError(f"unrecognised qubit_layout: {layout!r}")
    return int(m.group(1)), int(m.group(2))


def _to_bool(s: str) -> bool:
    return s == "True"


@dataclass(frozen=True)
class TCultivationRow:

    trial: int
    code_distance: int
    qubit_layout: tuple[int, int]
    placement: str
    n_qubits: int
    n_factories: int
    n_aods: int
    factory_physical_size: int
    trivial_return: bool
    decompose_move: bool
    redistribute_stage1_success: bool
    total_depth: float
    n_cnot: int
    n_cnot_teleportation: int
    n_h: int
    n_s: int
    n_t: int
    n_se_q: int
    fidelity_total: float
    fidelity_cnot: float
    fidelity_h: float
    fidelity_idle: float
    fidelity_of_t_gate: float


def load_results(path: Path = RESULTS_CSV) -> list[TCultivationRow]:

    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append(TCultivationRow(
                trial=int(r["trial"]),
                code_distance=int(r["code_distance"]),
                qubit_layout=parse_qubit_layout(r["qubit_layout"]),
                placement=r["placement"],
                n_qubits=int(r["n_qubits"]),
                n_factories=int(r["n_factories"]),
                n_aods=int(r["n_aods"]),
                factory_physical_size=int(r["factory_physical_size"]),
                trivial_return=_to_bool(r["trivial_return"]),
                decompose_move=_to_bool(r["decompose_move"]),
                redistribute_stage1_success=_to_bool(r["redistribute_stage1_success"]),
                total_depth=float(r["total_depth"]),
                n_cnot=int(r["n_cnot"]),
                n_cnot_teleportation=int(r["n_cnot_teleportation"]),
                n_h=int(r["n_h"]),
                n_s=int(r["n_s"]),
                n_t=int(r["n_t"]),
                n_se_q=int(r["n_se_q"]),
                fidelity_total=float(r["fidelity_total"]),
                fidelity_cnot=float(r["fidelity_cnot"]),
                fidelity_h=float(r["fidelity_h"]),
                fidelity_idle=float(r["fidelity_idle"]),
                fidelity_of_t_gate=float(r["fidelity_of_t_gate"]),
            ))
    return rows


def filter_rows(
    rows: list[TCultivationRow], *,
    code_distance: int | None = None,
    qubit_layout: tuple[int, int] | None = None,
    n_aods: int | None = None,
    placement: str | None = None,
    trivial_return: bool | None = None,
    decompose_move: bool | None = None,
    redistribute_stage1_success: bool | None = None,
) -> list[TCultivationRow]:

    """
    Narrow the reference rows down to one FT-Weave configuration (code
    distance, qubit layout, AOD count, placement strategy, and the three
    optimization ablation flags) - typically down to the 10 trial rows
    for that exact config, which is what a NASim comparison run should
    be measured against.
    """

    def keep(r: TCultivationRow) -> bool:
        if code_distance is not None and r.code_distance != code_distance: return False
        if qubit_layout is not None and r.qubit_layout != qubit_layout: return False
        if n_aods is not None and r.n_aods != n_aods: return False
        if placement is not None and r.placement != placement: return False
        if trivial_return is not None and r.trivial_return != trivial_return: return False
        if decompose_move is not None and r.decompose_move != decompose_move: return False
        if redistribute_stage1_success is not None and r.redistribute_stage1_success != redistribute_stage1_success: return False
        return True

    return [r for r in rows if keep(r)]

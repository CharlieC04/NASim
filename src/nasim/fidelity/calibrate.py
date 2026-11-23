from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nasim.fidelity.circuits import gen_bicycle_memory_circuit, surface_code_memory_circuit
from nasim.fidelity.decoders import decode_gen_bicycle_xz_split, decode_batch_matching
from nasim.qldpc import GenBicycleCode

@dataclass
class CalibrationResult:

    p_phys: float
    rounds: int
    shots: int
    n_errors: int

    @property
    def logical_error_rate(self) -> float:
        return self.n_errors / self.shots

    @property
    def per_round_logical_error_rate(self) -> float:
        return 1.0 - (1.0 - self.logical_error_rate) ** (1.0 / self.rounds)

def calibrate_gen_bicycle_code(code: GenBicycleCode, *, p_phys: float, rounds: int,
                               shots: int | None = None, target_errors: int = 100,
                               max_shots: int = 20000, basis: str = "Z", fast: bool = True) -> CalibrationResult:

    """
    qSieve calibration, sit code idle for 'rounds' and decode with BP+OSD
    """

    basis_coord = 1 if basis == "Z" else 0
    circuit = gen_bicycle_memory_circuit(code, rounds=rounds, p=p_phys, basis=basis)
    sampler = circuit.compile_detector_sampler()
    decode_kwargs = dict(max_iter=30, osd_order=0) if fast else dict(max_iter=1000, osd_order=10)

    if shots is not None:
        detection_events, observable_flips = sampler.sample(shots=shots, separate_observables=True)
        predicted = decode_gen_bicycle_xz_split(circuit, detection_events, basis_coord=basis_coord, **decode_kwargs)
        n_errors = int(np.sum(np.any(predicted != observable_flips, axis=1)))
        return CalibrationResult(p_phys=p_phys, rounds=rounds, shots=shots, n_errors=n_errors)

    total_shots = 0
    total_errors = 0
    batch = 200
    while total_errors < target_errors and total_shots < max_shots:
        this_batch = min(batch, max_shots - total_shots)
        detection_events, observable_flips = sampler.sample(shots=this_batch, separate_observables=True)
        predicted = decode_gen_bicycle_xz_split(circuit, detection_events, basis_coord=basis_coord, **decode_kwargs)
        total_errors += int(np.sum(np.any(predicted != observable_flips, axis=1)))
        total_shots += this_batch

    return CalibrationResult(p_phys=p_phys, rounds=rounds, shots=total_shots, n_errors=total_errors)

def calibrate_surface_code(*, distance: int, p_phys: float, rounds: int, shots: int, basis: str = "Z") -> CalibrationResult:

    circuit = surface_code_memory_circuit(distance=distance, rounds=rounds, p=p_phys, basis=basis)
    dem = circuit.detector_error_model(decompose_errors=True)

    sampler = circuit.compile_detector_sampler()
    detection_events, observable_flips = sampler.sample(shots=shots, separate_observables=True)

    predicted = decode_batch_matching(dem, detection_events)
    n_errors = int(np.sum(np.any(predicted != observable_flips, axis=1)))
    return CalibrationResult(p_phys=p_phys, rounds=rounds, shots=shots, n_errors=n_errors)
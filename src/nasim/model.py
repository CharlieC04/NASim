from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nasim.config.loader import load_config

@dataclass(frozen=True)
class Model:

    name: str
    description: str
    dimensions: str

    # geometry
    layers: int
    layer_spacing_um: float
    gate_pair_dist_um: float

    # interaction
    blockade_radius_um: float

    #transport
    v_xy_um_per_us: float
    v_loss_um_per_us: float
    handover_time_us: float
    handover_fidelity: float
    move_profile: str

    # addressing
    addressing: dict[str, Any]

    # readout
    readout_fidelity: float
    readout_time_us: float

    # gates
    cz_fidelity: float
    cz_duration_us: float
    single_qubit_fidelity: float
    single_qubit_duration_us: float
    carrier_fidelity: float

    # coherence
    t2_s: float

    # magic
    cultivation_check_success_prob: float
    cultivation_check_cycles: float
    cultivation_num_subfactories: int
    cultivation_escape_success_prob: float
    cultivation_escape_cycles_by_distance: dict[int, float]
    cultivation_teleportation_success_prob: float
    cultivation_t_state_infidelity_target: float

    # Only run SE every N cycles for idle qubits (based on FT-Weave)
    logical_se_interval: int = 1

    @property
    def qec_round_duration_us(self) -> float:

        """
        4 seq CZ
        """

        return 4 * self.cz_duration_us

    def escape_cycles_for_distance(self, distance: int) -> float:

        table = self.cultivation_escape_cycles_by_distance
        if distance in table: return table[distance]
        if distance < min(table): return table[min(table)]

        raise ValueError("No escape cycle count for code distance")


    @classmethod
    def from_yaml(cls, path: str | Path) -> Model:

        data = load_config(path)

        geometry = data["geometry"]
        interaction = data["interaction"]
        transport = data["transport"]
        readout = data["readout"]
        gates = data["gates"]
        coherence = data["coherence"]
        cultivation = data["cultivation"]

        return cls(
            name=data["name"],
            description=data["description"].strip(),
            dimensions=data["dimensions"],
            layers=geometry["layers"],
            layer_spacing_um=geometry["layer_spacing_um"],
            gate_pair_dist_um=geometry["gate_pair_dist_um"],
            blockade_radius_um=interaction["blockade_radius_um"],
            v_xy_um_per_us=transport["v_xy_um_per_us"],
            v_loss_um_per_us=transport["v_loss_um_per_us"],
            handover_time_us=transport["handover_time_us"],
            handover_fidelity=transport["handover_fidelity"],
            move_profile=transport["move_profile"],
            addressing=data["addressing"],
            readout_fidelity=readout["fidelity"],
            readout_time_us=readout["time_us"],
            cz_fidelity=gates["cz_fidelity"],
            cz_duration_us=gates["cz_duration_us"],
            single_qubit_fidelity=gates["single_qubit_fidelity"],
            single_qubit_duration_us=gates["single_qubit_duration_us"],
            carrier_fidelity=gates["carrier_fidelity"],
            t2_s=coherence["t2_s"],
            cultivation_check_success_prob=cultivation["check_success_prob"],
            cultivation_check_cycles=cultivation["check_cycles"],
            cultivation_num_subfactories=cultivation["num_subfactories"],
            cultivation_escape_success_prob=cultivation["escape_success_prob"],
            cultivation_escape_cycles_by_distance = {
                int(d): float(v) for d, v in cultivation["escape_cycles_by_distance"].items()
            },
            cultivation_teleportation_success_prob=cultivation["teleportation_success_prob"],
            cultivation_t_state_infidelity_target=cultivation["t_state_infidelity_target"],
            logical_se_interval=int(data.get("logical_se_interval", 1))
        )
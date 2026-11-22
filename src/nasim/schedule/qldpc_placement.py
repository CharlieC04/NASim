from __future__ import annotations

from nasim.circuit import Circuit, GateType
from nasim.schedule.qldpc_memory import MemoryBlock, MemoryOccupancy

def _interference_weights(circuit: Circuit) -> dict[tuple[int, int], int]:

    weights: dict[tuple[int, int], int] = {}
    for stage in circuit.stages:
        for gate in stage:
            if gate.type is GateType.TWO_QUBIT:
                a, b = sorted(gate.qubits)
                weights[(a,b)] = weights.get((a,b), 0) + 1

    return weights

def assign_qubits_to_blocks(circuit: Circuit, block_ids: list[int], capacity_per_block: int) -> dict[int, int]:

    """
    Use colour mapping based on 2q interactions
    """

    if circuit.num_qubits > len(block_ids) * capacity_per_block:
        raise ValueError("circuit does not fit in blocks")

    weights = _interference_weights(circuit)

    neighbour_weight: dict[int, dict[int, int]] = {q: {} for q in range(circuit.num_qubits)}
    degree: dict[int, int] = {q: 0 for q in range(circuit.num_qubits)}
    for (a, b), w in weights.items():
        neighbour_weight[a][b] = w
        neighbour_weight[b][a] = w
        degree[a] += w
        degree[b] += w

    order = sorted(range(circuit.num_qubits), key=lambda q: -degree[q])

    assignment: dict[int, int] = {}
    block_count: dict[int, int] = {b: 0 for b in block_ids}

    for q in order:
        best_block = None
        best_cost = None
        for b in block_ids:
            if block_count[b] >= capacity_per_block: continue

            cost = sum(w for nb, w in neighbour_weight[q].items() if assignment.get(nb) == b)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_block = b

        if best_block is None: raise ValueError("no block has free capacity")

        assignment[q] = best_block
        block_count[best_block] += 1

    return assignment

def build_initial_memory_state(circuit: Circuit, blocks: dict[int, MemoryBlock]) -> tuple[dict[int, int], 
                                                                                          dict[int, MemoryOccupancy]]:

    block_ids = list(blocks.keys())
    if not block_ids: raise ValueError("no memory blocks")

    code = blocks[block_ids[0]].code
    capacity = code.k

    assignment = assign_qubits_to_blocks(circuit, block_ids, capacity)

    occupancies = {b: MemoryOccupancy.empty(blocks[b]) for b in block_ids}
    for q, b in assignment.items():
        slot = occupancies[b].free_slots()[0]
        occupancies[b].store(slot, q)

    return assignment, occupancies
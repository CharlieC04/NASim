from __future__ import annotations

import math
from itertools import combinations

QSIEVE_SPACING_UM = 5.0
QSIEVE_ACCEL_UM_PER_US2 = 0.02

def movement_cost_us(delta_row: int, delta_col: int, *, spacing_um: float, accel: float) -> float:

    dx = abs(delta_col) * spacing_um
    dy = abs(delta_row) * spacing_um
    return math.sqrt(6 * dx / accel) + math.sqrt(6 * dy / accel)

def route_stops(ofs: list[tuple[int, int]], *, spacing_um: float, accel: float) -> tuple[list[tuple[int, int]], float]:

    """
    Exact tour that starts and ends at (0,0) and visits all required stops
    """

    points = [(0,0)] + sorted(set(ofs))
    n = len(points)

    def cost(i: int, j: int) -> float:
        dr = points[j][0] - points[i][0]
        dc = points[j][1] - points[i][1]
        return movement_cost_us(dr, dc, spacing_um=spacing_um, accel=accel)

    if n == 1: return [points[0], points[0]], 0.0

    others = list(range(1, n))
    dp: dict[tuple[int, int], tuple[float, list[int]]] = {}
    for idx, j in enumerate(others):
        dp[(1 << idx, j)] = (cost(0, j), [0, j])

    for size in range(2, len(others) + 1):
        for subset in combinations(others, size):
            mask = sum(1 << others.index(x) for x in subset)
            for j in subset:
                prev_mask = mask & ~(1 << others.index(j))
                best = None
                for k in subset:
                    if k == j: continue
                    prev = dp.get((prev_mask, k))
                    if prev is None: continue
                    c = prev[0] + cost(k, j)
                    if best is None or c < best[0]:
                        best = (c, prev[1] + [j])
                if best is not None:
                    dp[(mask, j)] = best

    full_mask = (1 << len(others)) - 1
    best_total, best_path = None, None
    for j in others:
        entry = dp.get((full_mask, j))
        if entry is None: continue
        total = entry[0] + cost(j, 0)
        if best_total is None or total < best_total:
            best_total, best_path = total, entry[1] + [0]

    assert best_path is not None and best_total is not None
    return [points[i] for i in best_path], best_total
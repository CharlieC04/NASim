from __future__ import annotations

import math
from typing import NamedTuple

class Position(NamedTuple):

    x: float
    y: float
    z: float = 0.0

def distance(a: Position, b: Position) -> float:

    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

def _rank_gf2(matrix: np.ndarray) -> int:

    """
    Rank of bin mat over GF(2)
    """

    m = matrix.copy().astype(np.uint8) % 2
    rows, cols = m.shape
    rank = 0

    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if m[r, col]:
                pivot = r
                break

        if pivot is None: continue

        m[[rank, pivot]] = m[[pivot, rank]]
        for r in range(rows):
            if r != rank and m[r, col]: m[r] ^= m[rank]

        rank += 1
        if rank == rows: break

    return rank

def _perm_power(size: int, power: int) -> np.ndarray:

    """
    Perm matrix mapping row i to col (i+p) mod size
    """

    mat = np.zeros((size, size), dtype=np.uint8)
    for i in range(size): mat[i, (i+power) % size] = 1
    return mat

@dataclass(frozen=True)
class BicycleTerm:

    """
    One mono, x^p y^q
    """

    p: int
    q: int

@dataclass(frozen=True)
class GenBicycleCode:

    """
    A BB code built from a(x,y), b(x,y)
    """

    l: int
    m: int
    a_terms: tuple[BicycleTerm, ...]
    b_terms: tuple[BicycleTerm, ...]
    name: str = ""

    @property
    def lm(self) -> int:
        return self.l * self.m

    def _term_matrix(self, term: BicycleTerm) -> np.ndarray:
        return np.kron(_perm_power(self.l, term.p), _perm_power(self.m, term.q))

    def _sum_terms(self, terms: tuple[BicycleTerm, ...]) -> np.ndarray:
        acc = np.zeros((self.lm, self.lm), dtype=np.uint8)
        for t in terms: acc = (acc + self._term_matrix(t)) % 2
        return acc

    @property
    def A(self) -> np.ndarray:
        return self._sum_terms(self.a_terms)

    @property
    def B(self) -> np.ndarray:
        return self._sum_terms(self.b_terms)

    @property
    def Hx(self) -> np.ndarray:
        return np.concatenate([self.A, self.B], axis=1)

    @property
    def Hz(self) -> np.ndarray:
        return np.concatenate([self.B.T, self.A.T], axis=1) % 2

    @property
    def n(self) -> int:
        return 2 * self.lm

    @property
    def k(self) -> int:
        return self.n - _rank_gf2(self.Hx) - _rank_gf2(self.Hz)

    @property
    def check_weight(self) -> int:
        return len(self.a_terms) + len(self.b_terms)

    def grid_position(self, index: int) -> tuple[int, int]:
        if not (0 <= index < self.lm):
            raise ValueError("index out of range")

        return (index % self.m, index // self.m)

BB_144_12_12 = GenBicycleCode(
    l=12, m=6,
    a_terms=(BicycleTerm(p=0, q=1), BicycleTerm(p=0, q=2), BicycleTerm(p=3, q=0)),
    b_terms=(BicycleTerm(p=0, q=3), BicycleTerm(p=1, q=0), BicycleTerm(p=2, q=0)),
    name="[[144,12,12]]"
)
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

def _gf2_rref_with_pivots(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:

    m = matrix.copy().astype(np.uint8) % 2
    rows, cols = m.shape
    pivot_cols: list[int] = []
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

        pivot_cols.append(col)
        rank += 1
        if rank == rows: break

    return m[:rank], pivot_cols

def _gf2_nullspace(matrix: np.ndarray) -> list[np.ndarray]:

    rref, pivot_cols = _gf2_rref_with_pivots(matrix)
    cols = matrix.shape[1]
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(cols) if c not in pivot_set]

    basis = []
    for free in free_cols:
        vec = np.zeros(cols, dtype=np.uint8)
        vec[free] = 1
        for i, pc in enumerate(pivot_cols):
            if rref[i, free]:
                vec[pc] = 1
        basis.append(vec)

    return basis

def _gf2_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint8) @ b.astype(np.uint8)) % 2

def _gf2_inverse(matrix: np.ndarray) -> np.ndarray:

    n = matrix.shape[0]
    aug = np.concatenate([matrix.copy().astype(np.uint8) % 2, np.eye(n, dtype=np.uint8)], axis=1)
    rank = 0

    for col in range(n):
        pivot = None
        for r in range(rank, n):
            if aug[r, col]:
                pivot = r
                break
        if pivot is None: raise ValueError("matrix singular")

        aug[[rank, pivot]] = aug[[pivot, rank]]
        for r in range(n):
            if r != rank and aug[r, col]: aug[r] ^= aug[rank]
        rank += 1

    return aug[:, n:]

def _logical_reps(ker_of: np.ndarray, quotient_by: np.ndarray, count: int) -> np.ndarray:

    """
    'count' indep repr of ker(ker_of) modulo rowspan(quot_by)
    """

    kernel_basis = _gf2_nullspace(ker_of)
    stab_rref, stab_pivots = _gf2_rref_with_pivots(quotient_by)

    basis_rows = list(stab_rref)
    pivot_cols = list(stab_pivots)
    reps = []

    for vec in kernel_basis:
        reduced = vec.copy()
        for row, pc in zip(basis_rows, pivot_cols):
            if reduced[pc]: reduced ^= row

        if np.any(reduced):
            pivot_col = int(np.nonzero(reduced)[0][0])
            basis_rows.append(reduced)
            pivot_cols.append(pivot_col)
            reps.append(vec)
            if len(reps) == count: break

    if len(reps) != count: raise ValueError("different reps to count")

    return np.array(reps, dtype=np.uint8)

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

    def _negate_term(self, term: BicycleTerm) -> BicycleTerm:
        return BicycleTerm(p=(-term.p) % self.l, q=(-term.q) % self.m)

    def _term_ofs(self, check_row: int, check_col: int, p: int, q: int) -> tuple[int, int]:

        p_mod, q_mod = p % self.l, q % self.m
        row_ofs = q_mod if (check_row + q_mod) < self.m else q_mod - self.m
        col_ofs = p_mod if (check_col + p_mod) < self.l else p_mod - self.l
        return (row_ofs, col_ofs)

    def systolic_offsets(self, hx: bool = True) -> list[tuple[int, int]]:

        """
        Offsets from check qubit to data qubits, checking for systolic movement
        """

        terms = self.a_terms + self.b_terms if hx else tuple(
            self._negate_term(t) for t in self.b_terms + self.a_terms
        )

        ofs: set[tuple[int, int]] = set()
        for term in terms:
            for check_idx in range(self.lm):
                check_row, check_col = self.grid_position(check_idx)
                ofs.add(self._term_ofs(check_row, check_col, term.p, term.q))

        return sorted(ofs)

    def logical_operators(self) -> list[tuple[frozenset[int], frozenset[int]]]:

        """
        For each of k log qubits, data q support of repr
        """

        z_reps = _logical_reps(self.Hx, self.Hz, self.k)
        x_reps = _logical_reps(self.Hz, self.Hx, self.k)

        overlap = _gf2_matmul(x_reps, z_reps.T)
        overlap_inv = _gf2_inverse(overlap)
        x_paired = _gf2_matmul(overlap_inv, x_reps)

        return [
            (
                frozenset(np.nonzero(x_paired[i])[0].tolist()),
                frozenset(np.nonzero(z_reps[i])[0].tolist()),
            )
            for i in range(self.k)
        ]

BB_144_12_12 = GenBicycleCode(
    l=12, m=6,
    a_terms=(BicycleTerm(p=0, q=1), BicycleTerm(p=0, q=2), BicycleTerm(p=3, q=0)),
    b_terms=(BicycleTerm(p=0, q=3), BicycleTerm(p=1, q=0), BicycleTerm(p=2, q=0)),
    name="[[144,12,12]]"
)
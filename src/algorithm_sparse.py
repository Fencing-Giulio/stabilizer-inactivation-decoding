
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import random

ERASURE = -1
INT = np.int8


@dataclass
class SparseMatrix:
    rows: List[set[int]]
    n_cols: int

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "SparseMatrix":
        arr = (arr % 2).astype(INT)
        rows = [set(np.flatnonzero(row)) for row in arr]
        return cls(rows, arr.shape[1])

    def copy(self) -> "SparseMatrix":
        return SparseMatrix([set(r) for r in self.rows], self.n_cols)

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.rows), self.n_cols

    def row_count(self) -> int:
        return len(self.rows)

    def swap_rows(self, i: int, j: int) -> None:
        if i == j:
            return
        self.rows[i], self.rows[j] = self.rows[j], self.rows[i]

    def move_row_to_front(self, row_idx: int, front_idx: int) -> None:
        if row_idx == front_idx:
            return
        row = self.rows.pop(row_idx)
        self.rows.insert(front_idx, row)

    def clear_row(self, idx: int) -> None:
        self.rows[idx] = set()

    def xor_rows(self, target: int, source: int) -> None:
        if target == source:
            return
        self.rows[target] = self.rows[target] ^ self.rows[source]

    def swap_columns(self, c1: int, c2: int) -> None:
        if c1 == c2:
            return
        for row in self.rows:
            has_c1 = c1 in row
            has_c2 = c2 in row
            if has_c1 and has_c2:
                continue
            if has_c1:
                row.remove(c1)
                row.add(c2)
            elif has_c2:
                row.remove(c2)
                row.add(c1)

    def permuted(self, perm: Sequence[int]) -> "SparseMatrix":
        if len(perm) != self.n_cols:
            raise ValueError("Permutation length mismatch")
        inverse = {old: new for new, old in enumerate(perm)}
        new_rows = [set(inverse[col] for col in row) for row in self.rows]
        return SparseMatrix(new_rows, self.n_cols)

    def remove_columns(self, cols_to_remove: Sequence[int]) -> "SparseMatrix":
        remove_set = set(cols_to_remove)
        mapping = {}
        new_idx = 0
        for old in range(self.n_cols):
            if old in remove_set:
                continue
            mapping[old] = new_idx
            new_idx += 1
        new_rows = [
            {mapping[col] for col in row if col in mapping}
            for row in self.rows
        ]
        return SparseMatrix(new_rows, new_idx)

    def append_column(self, column: Iterable[int]) -> "SparseMatrix":
        new_rows = [set(r) for r in self.rows]
        new_col = self.n_cols
        for idx, val in enumerate(column):
            if val % 2:
                new_rows[idx].add(new_col)
        return SparseMatrix(new_rows, self.n_cols + 1)

    def to_numpy(self) -> np.ndarray:
        arr = np.zeros(self.shape, dtype=INT)
        for i, row in enumerate(self.rows):
            for col in row:
                arr[i, col] = 1
        return arr

    def row_weight(self, row_idx: int, *, start: int = 0, end: Optional[int] = None) -> int:
        if end is None:
            end = self.n_cols
        return sum(1 for col in self.rows[row_idx] if start <= col < end)

    def first_column(self, row_idx: int, *, start: int = 0, end: Optional[int] = None) -> Optional[int]:
        if end is None:
            end = self.n_cols
        cols = [col for col in self.rows[row_idx] if start <= col < end]
        return min(cols) if cols else None

    def has_column(self, row_idx: int, column: int) -> bool:
        return column in self.rows[row_idx]

    def rows_with_column(self, column: int, *, start_row: int = 0) -> List[int]:
        return [idx for idx in range(start_row, self.row_count()) if column in self.rows[idx]]


def as_int(arr: Iterable[int]) -> np.ndarray:
    return np.asarray(arr, dtype=INT)


def ensure_sparse(matrix: np.ndarray | SparseMatrix) -> SparseMatrix:
    if isinstance(matrix, SparseMatrix):
        return matrix.copy()
    return SparseMatrix.from_numpy(matrix)


def normalize_boundary(sep: int, n_cols: int) -> int:
    if sep >= 0:
        return min(sep, n_cols)
    return max(0, n_cols + sep)


def row_left_weight(matrix: SparseMatrix, row_idx: int, sep: int) -> int:
    boundary = normalize_boundary(sep, matrix.n_cols)
    return matrix.row_weight(row_idx, end=boundary)


def first_left_column(matrix: SparseMatrix, row_idx: int, sep: int) -> Optional[int]:
    boundary = normalize_boundary(sep, matrix.n_cols)
    return matrix.first_column(row_idx, end=boundary)


def row_has_left(matrix: SparseMatrix, row_idx: int, sep: int) -> bool:
    boundary = normalize_boundary(sep, matrix.n_cols)
    return any(col < boundary for col in matrix.rows[row_idx])


def partition_by_erasures(msg: np.ndarray, H: np.ndarray | SparseMatrix):
    matrix = ensure_sparse(H)
    non_er_cols = [i for i, v in enumerate(msg) if v != ERASURE]
    erasure_cols = [i for i, v in enumerate(msg) if v == ERASURE]
    perm = np.array(non_er_cols + erasure_cols, dtype=int)
    Hp = matrix.permuted(perm.tolist())
    boundary = len(non_er_cols)
    return Hp, perm, boundary


def move_zero_left_rows_to_bottom(
    H: SparseMatrix, boundary: int, start_row: int = 0
) -> SparseMatrix:
    matrix = H.copy()
    head = matrix.rows[:start_row]
    tail = matrix.rows[start_row:]
    non_zero_rows = []
    zero_rows = []
    for idx, row in enumerate(tail):
        actual_idx = start_row + idx
        if any(col < boundary for col in row):
            non_zero_rows.append(actual_idx)
        else:
            zero_rows.append(actual_idx)
    reordered = head + [matrix.rows[i] for i in non_zero_rows] + [matrix.rows[i] for i in zero_rows]
    return SparseMatrix([set(r) for r in reordered], matrix.n_cols)


def sweep_one_pivot(
    H: SparseMatrix,
    msg: np.ndarray,
    perm: np.ndarray,
    boundary: int,
    start_row: int,
    *,
    rng=np.random,
):
    matrix = H.copy()
    left_zero_candidates = []
    for offset in range(start_row, matrix.row_count()):
        row = matrix.rows[offset]
        has_left = any(col < boundary for col in row)
        has_any = bool(row)
        if not has_left and has_any:
            left_zero_candidates.append(offset)
    if not left_zero_candidates:
        return False, matrix, msg, perm, boundary

    pivot_r = left_zero_candidates[0]
    pivot_candidates = [col for col in sorted(matrix.rows[pivot_r]) if col >= boundary]
    if not pivot_candidates:
        return False, matrix, msg, perm, boundary
    pivot_c = pivot_candidates[0]

    if pivot_c != boundary:
        matrix.swap_columns(pivot_c, boundary)
        perm[pivot_c], perm[boundary] = perm[boundary], perm[pivot_c]

    boundary += 1
    target_c = boundary - 1

    if pivot_r != start_row:
        matrix.swap_rows(pivot_r, start_row)

    orig_col = perm[target_c]
    if msg[orig_col] == ERASURE:
        msg[orig_col] = 0

    return True, matrix, msg, perm, boundary


def peel_left_side(H: SparseMatrix, boundary: int, start_row: int):
    matrix = H.copy()
    ignore = start_row
    while True:
        if boundary == 0 or ignore >= matrix.row_count():
            break
        pivot_r = None
        for idx in range(ignore, matrix.row_count()):
            if matrix.row_weight(idx, end=boundary) == 1:
                pivot_r = idx
                break
        if pivot_r is None:
            break
        pivot_c = matrix.first_column(pivot_r, end=boundary)
        assert pivot_c is not None
        for row_idx in range(ignore, matrix.row_count()):
            if row_idx != pivot_r and matrix.has_column(row_idx, pivot_c):
                matrix.xor_rows(row_idx, pivot_r)
        if pivot_r != ignore:
            matrix.swap_rows(pivot_r, ignore)
        ignore += 1
    return matrix, ignore


def squeeze_multiweight_left_columns(
    H: np.ndarray | SparseMatrix,
    boundary: int,
    start_row: int = 0,
) -> SparseMatrix:
    matrix = ensure_sparse(H)
    while True:
        weights = {}
        for row_idx in range(start_row, matrix.row_count()):
            for col in matrix.rows[row_idx]:
                if col < boundary:
                    weights[col] = weights.get(col, 0) + 1
        bad_cols = [col for col, w in sorted(weights.items()) if w > 1]
        if not bad_cols:
            break
        c = bad_cols[0]
        abs_rows = [
            row_idx
            for row_idx in range(start_row, matrix.row_count())
            if c in matrix.rows[row_idx]
        ]
        if len(abs_rows) < 2:
            break
        r0, r1 = abs_rows[0], abs_rows[1]
        matrix.xor_rows(r0, r1)
        matrix.clear_row(r1)
    return matrix


def structured_decode(msg: np.ndarray, H: np.ndarray | SparseMatrix, *, rng=np.random):
    Hwork, perm, boundary = partition_by_erasures(msg, H)
    pivot_total = 0
    row_ptr = 0

    while True:
        Hwork = move_zero_left_rows_to_bottom(Hwork, boundary, row_ptr)
        Hwork, row_ptr = peel_left_side(Hwork, boundary, row_ptr)
        pivot_found, Hwork, msg, perm, boundary = sweep_one_pivot(
            Hwork, msg, perm, boundary, row_ptr, rng=rng
        )
        if not pivot_found:
            break
        pivot_total += 1
        

    Hwork = squeeze_multiweight_left_columns(Hwork, boundary, start_row=row_ptr)

    while True:
        Hwork = move_zero_left_rows_to_bottom(Hwork, boundary, row_ptr)
        pivot_found, Hwork, msg, perm, boundary = sweep_one_pivot(
            Hwork, msg, perm, boundary, row_ptr, rng=rng
        )
        if not pivot_found:
            break
        pivot_total += 1
        Hwork, row_ptr = peel_left_side(Hwork, boundary, row_ptr)

    return msg, pivot_total


def syndrome(H: np.ndarray | SparseMatrix, msg: np.ndarray):
    matrix = ensure_sparse(H)
    temp = np.where(msg == ERASURE, np.random.randint(0, 2, size=msg.size), msg)
    s_values = []
    for row in matrix.rows:
        parity = 0
        for col in row:
            parity ^= int(temp[col] & 1)
        s_values.append(parity)
    s = np.array(s_values, dtype=INT)
    return s, temp.astype(INT)


def agm_matrix_generator(H: np.ndarray | SparseMatrix, s: np.ndarray):
    matrix = ensure_sparse(H)
    s = as_int(s).reshape(-1, 1)
    return matrix.append_column(s.flatten()), -1


def removal(msg: np.ndarray, H: SparseMatrix):
    idx_to_remove = [i for i, v in enumerate(msg) if v != ERASURE]
    return H.remove_columns(idx_to_remove)


def create_tracking_vector(msg: np.ndarray):
    return [i for i, v in enumerate(msg) if v == ERASURE] + [-1]


def one_bit_per_row(H_aug: SparseMatrix, sep: int, ignore: int):
    dangling = []
    for idx in range(ignore, H_aug.row_count()):
        if row_left_weight(H_aug, idx, sep) == 1:
            dangling.append(idx - ignore)
    return dangling, bool(dangling)


def guess(matrix: SparseMatrix, sep: int, tracking_vector: list):
    n_cols = matrix.n_cols
    if n_cols <= 1:
        return matrix, sep, tracking_vector
    insert_at = sep % n_cols
    perm = list(range(n_cols))
    boundary = normalize_boundary(sep, n_cols)
    if boundary <= 0:
        return matrix, sep, tracking_vector
    col_weights = [0] * boundary
    for row in matrix.rows:
        for col in row:
            if col < boundary:
                col_weights[col] += 1
    col_idx = int(np.argmax(col_weights))
    col0 = perm.pop(col_idx)
    perm.insert(insert_at - 1, col0)
    new_matrix = matrix.permuted(perm)
    tracking_vector = list(np.array(tracking_vector, dtype=object)[perm])
    return new_matrix, sep - 1, tracking_vector


def move_zero_rows_to_bottom(H: SparseMatrix, sep: int):
    non_zero = []
    zero = []
    for idx in range(H.row_count()):
        if row_has_left(H, idx, sep):
            non_zero.append(idx)
        else:
            zero.append(idx)
    reordered = [H.rows[i] for i in non_zero + zero]
    return SparseMatrix([set(r) for r in reordered], H.n_cols)


def find_horizontal_boundary(A: SparseMatrix, col_boundary: int) -> int:
    rb = 0
    for idx in range(A.row_count()):
        if row_has_left(A, idx, col_boundary):
            rb += 1
        else:
            break
    return rb


def solve_gf2_random(A_np: np.ndarray, b_np: np.ndarray, rng=random):
    A, b = as_int(A_np) % 2, as_int(b_np).reshape(-1, 1) % 2
    aug = np.hstack((A, b))
    m, n1 = aug.shape
    n = n1 - 1
    pivots = []
    r = 0
    for c in range(n):
        swap = np.where(aug[r:, c])[0]
        if swap.size == 0:
            continue
        swap = swap[0] + r
        aug[[r, swap]] = aug[[swap, r]]
        pivots.append(c)
        for i in range(m):
            if i != r and aug[i, c]:
                aug[i] ^= aug[r]
        r += 1
        if r == m:
            break

    if np.any((aug[:, :n] == 0).all(1) & (aug[:, -1] == 1)):
        raise ValueError("No GF(2) solution")

    x = np.zeros(n, dtype=np.uint8)
    free_cols = [c for c in range(n) if c not in pivots]
    for col in free_cols:
        x[col] = rng.randint(0, 1)

    for row_idx in reversed(range(len(pivots))):
        col = pivots[row_idx]
        s = (aug[row_idx, col + 1 : n] & x[col + 1 :]).sum() & 1
        x[col] = aug[row_idx, -1] ^ s
    return x.tolist()


def solver(H: SparseMatrix, sep1: int, sep2: int) -> np.ndarray:
    H_np = H.to_numpy()
    G = H_np[sep2:, sep1:]
    permutation = H_np[:sep2, :sep1].T

    A, b = G[:, :-1], G[:, -1]
    guesses = np.array(solve_gf2_random(A, b))
    if guesses.size == 0:
        guesses = np.random.randint(0, 2, size=abs(sep1 + 1))
    guesses = np.append(guesses, 1)

    E = H_np[:sep2, sep1:]
    result = (E @ guesses) % 2
    result = permutation @ result
    return np.concatenate((result, guesses))


def sort_and_fill(tracker: list[int], unsolved_answer: np.ndarray, msg: np.ndarray) -> np.ndarray:
    """
    tracker[:-1] = original erased column indices, in the SAME order
                   the inactivation logic used for its left variables.
    unsolved_answer[:-1] = recovered values for those vars
    unsolved_answer[-1]  = RHS / syndrome bit (keep if you need it)
    """
    msg_filled = msg.copy()

    values = unsolved_answer[:-1]

    for k, col_idx in enumerate(tracker[:-1]):
        msg_filled[col_idx] = int(values[k] & 1)

    return msg_filled


def inactivation_decoding(
    H_er: SparseMatrix,
    tracking: list,
    sep1: int,
    msg: np.ndarray,
    *,
    guess_cap: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    matrix = H_er.copy()
    ignore = 0
    guesses_done = 0
    while True:
        flag = True
        while flag:
            dangling_rows, flag = one_bit_per_row(matrix, sep1, ignore)
            if not dangling_rows:
                break
            pivot_idx = ignore + dangling_rows[0]
            col_idx = first_left_column(matrix, pivot_idx, sep1)
            assert col_idx is not None
            for j in range(ignore, matrix.row_count()):
                if j != pivot_idx and matrix.has_column(j, col_idx):
                    matrix.xor_rows(j, pivot_idx)
            matrix.move_row_to_front(pivot_idx, ignore)
            ignore += 1

        if all(row_left_weight(matrix, r, sep1) <= 1 for r in range(matrix.row_count())):
            break

        if guess_cap is not None and guesses_done >= guess_cap:
            break

        matrix, sep1, tracking = guess(matrix, sep1, tracking)
        guesses_done += 1

    matrix = move_zero_rows_to_bottom(matrix, sep1)
    sep2 = find_horizontal_boundary(matrix, sep1)
    unsolved = solver(matrix, sep1, sep2)
    final_answer = sort_and_fill(tracking, unsolved, msg)
    return final_answer, guesses_done


def hz_peel_only(Hz: np.ndarray | SparseMatrix, msg: np.ndarray, s):
    aug_matrix, sep1 = agm_matrix_generator(Hz, s)
    H_er = removal(msg, aug_matrix)
    tracking = create_tracking_vector(msg)

    matrix = H_er.copy()
    ignore = 0
    solved: dict[int, int] = {}

    while True:
        dangling_rows, flag = one_bit_per_row(matrix, sep1, ignore)
        if not flag:
            break
        pivot_idx = ignore + dangling_rows[0]
        col_idx = first_left_column(matrix, pivot_idx, sep1)
        assert col_idx is not None
        val = int(matrix.has_column(pivot_idx, matrix.n_cols - 1))
        sym = tracking[col_idx]
        if sym != -1:
            solved[sym] = val

        for j in range(ignore, matrix.row_count()):
            if j != pivot_idx and matrix.has_column(j, col_idx):
                matrix.xor_rows(j, pivot_idx)
        matrix.move_row_to_front(pivot_idx, ignore)
        ignore += 1

    corr = msg.copy()
    for k, v in solved.items():
        corr[k] = v

    return corr


def build_hz_system_and_inactivate(
    Hz: np.ndarray | SparseMatrix,
    msg: np.ndarray,
    s,
    *,
    guess_cap: Optional[int] = None,
):
    aug_matrix, sep1 = agm_matrix_generator(Hz, s)
    H_er = removal(msg, aug_matrix)
    tracking = create_tracking_vector(msg)
    final_corr, guesses = inactivation_decoding(
        H_er, tracking, sep1, msg, guess_cap=guess_cap
    )
    return final_corr, guesses

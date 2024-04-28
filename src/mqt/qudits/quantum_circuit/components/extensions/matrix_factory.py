from __future__ import annotations

import itertools
import operator
from functools import reduce

import numpy as np


class MatrixFactory:
    def __init__(self, gate, identities_flag) -> None:
        self.gate = gate
        self.ids = identities_flag

    def generate_matrix(self):
        lines = self.gate.reference_lines.copy()
        circuit = self.gate.parent_circuit
        ref_slice = list(range(min(lines), max(lines) + 1))
        dimensions_slice = circuit.dimensions[min(lines) : max(lines) + 1]
        matrix = self.gate.__array__()
        if self.gate.dagger:
            matrix = matrix.conj().T

        control_info = self.gate.control_info["controls"]

        if control_info:
            controls = control_info.indices
            ctrl_levs = control_info.ctrl_states
            # preferably only CONTROLLED-One qudit gates to be made as multi-controlled, it is still a low level
            # control library
            matrix = MatrixFactory.apply_identites_and_controls(
                matrix, self.gate._target_qudits, dimensions_slice, ref_slice, controls, ctrl_levs
            )
        elif self.ids > 0:
            matrix = MatrixFactory.apply_identites_and_controls(
                matrix, self.gate._target_qudits, dimensions_slice, ref_slice
            )

        if self.ids >= 2:
            matrix = MatrixFactory.wrap_in_identities(matrix, lines, circuit.dimensions)

        return matrix

    @classmethod
    def apply_identites_and_controls(
        cls, matrix, qudits_applied, dimensions, ref_lines, controls=None, controls_levels=None
    ):
        # dimensions = list(reversed(dimensions))
        # Convert qudits_applied and dimensions to lists if they are not already
        qudits_applied = [qudits_applied] if isinstance(qudits_applied, int) else qudits_applied
        qudits_applied = qudits_applied.copy()
        qudits_applied.sort()
        dimensions = [dimensions] if isinstance(dimensions, int) else dimensions
        if len(dimensions) == 0:
            msg = "Dimensions cannot be an empty list"
            raise ValueError(msg)
        if len(qudits_applied) == len(ref_lines) and controls is None:
            return matrix

        single_site_logics = []
        og_states_space = []
        og_state_to_index = {}

        if len(qudits_applied) == 1:
            single_site_logics.append(list(range(dimensions[qudits_applied[0]])))
        else:
            for d in list(operator.itemgetter(*qudits_applied)(dimensions)):
                single_site_logics.append(list(range(d)))

        for element in itertools.product(*single_site_logics):
            og_states_space.append(list(element))

        for i in range(len(og_states_space)):
            og_state_to_index[tuple(og_states_space[i])] = i

        global_single_site_logics = []
        global_states_space = []
        global_index_to_state = {}
        for d in dimensions:
            global_single_site_logics.append(list(range(d)))

        for element in itertools.product(*global_single_site_logics):
            global_states_space.append(list(element))

        for i in range(len(global_states_space)):
            global_index_to_state[i] = global_states_space[i]

        result = np.identity(reduce(operator.mul, dimensions, 1), dtype="complex")

        for r in range(result.shape[0]):
            for c in range(result.shape[1]):
                if controls is not None:
                    extract_r = operator.itemgetter(*controls)(global_index_to_state[r])
                    extract_c = operator.itemgetter(*controls)(global_index_to_state[c])
                    if isinstance(extract_r, int):
                        extract_r = [extract_r]
                        extract_c = [extract_c]
                    if list(extract_r) == controls_levels and extract_r == extract_c:
                        rest_of_indices = set(ref_lines) - set(qudits_applied) - set(controls)
                        if not rest_of_indices or operator.itemgetter(*rest_of_indices)(
                            global_index_to_state[r]
                        ) == operator.itemgetter(*rest_of_indices)(global_index_to_state[c]):
                            og_row_key = operator.itemgetter(*qudits_applied)(global_index_to_state[r])
                            og_col_key = operator.itemgetter(*qudits_applied)(global_index_to_state[c])
                            if isinstance(og_row_key, int):
                                og_row_key = (og_row_key,)
                            if isinstance(og_col_key, int):
                                og_col_key = (og_col_key,)
                            matrix_row = og_state_to_index[tuple(og_row_key)]
                            matrix_col = og_state_to_index[tuple(og_col_key)]
                            value = matrix[matrix_row, matrix_col]
                            result[r, c] = value

                else:
                    rest_of_indices = set(ref_lines) - set(qudits_applied)
                    if not rest_of_indices or operator.itemgetter(*rest_of_indices)(
                        global_index_to_state[r]
                    ) == operator.itemgetter(*rest_of_indices)(global_index_to_state[c]):
                        og_row_key = operator.itemgetter(*qudits_applied)(global_index_to_state[r])
                        og_col_key = operator.itemgetter(*qudits_applied)(global_index_to_state[c])
                        if isinstance(og_row_key, int):
                            og_row_key = (og_row_key,)
                        if isinstance(og_col_key, int):
                            og_col_key = (og_col_key,)
                        matrix_row = og_state_to_index[tuple(og_row_key)]
                        matrix_col = og_state_to_index[tuple(og_col_key)]
                        value = matrix[matrix_row, matrix_col]
                        result[r, c] = value

        return result

    @classmethod
    def wrap_in_identities(cls, matrix, indices, sizes):
        indices.sort()
        if any(index >= len(sizes) for index in indices):
            msg = "Index out of range"
            raise ValueError(msg)

        i = 0
        result = np.identity(sizes[i])
        while i < len(sizes):
            if i == indices[0]:
                result = matrix if i == 0 else np.kron(result, matrix)
            elif (i < indices[0] and i != 0) or i > indices[-1]:
                result = np.kron(result, np.identity(sizes[i]))

            i += 1

        return result


def from_dirac_to_basis(vec, d):  # |00> -> [1,0,...,0] -> len() == other_size**2
    if isinstance(d, int):
        d = [d] * len(vec)

    basis_vecs = []
    for i, basis in enumerate(vec):
        temp = [0] * d[i]
        temp[basis] = 1
        basis_vecs.append(temp)

    ret = basis_vecs[0]
    for e_i in range(1, len(basis_vecs)):
        ret = np.kron(np.array(ret), np.array(basis_vecs[e_i]))

    return ret


def calculate_q0_q1(lev, dim):
    q1 = lev % dim
    q0 = (lev - q1) // dim

    return q0, q1


def insert_at(big_arr, pos, to_insert_arr):
    """Quite a forceful way of embedding a parameters into big_arr."""
    x1 = pos[0]
    y1 = pos[1]
    x2 = x1 + to_insert_arr.shape[0]
    y2 = y1 + to_insert_arr.shape[1]

    assert x2 <= big_arr.shape[0], "the position will make the small parameters exceed the boundaries at x"
    assert y2 <= big_arr.shape[1], "the position will make the small parameters exceed the boundaries at y"

    big_arr[x1:x2, y1:y2] = to_insert_arr

    return big_arr

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ComplexArray = NDArray[np.complex128]


def on1(gate: ComplexArray, other_size: int) -> ComplexArray:
    return np.kron(np.identity(other_size, dtype=np.complex128), gate)


def on0(gate: ComplexArray, other_size: int) -> ComplexArray:
    return np.kron(gate, np.identity(other_size, dtype=np.complex128))


def gate_expand_to_circuit(
        gate: ComplexArray,
        circuits_size: int,
        target: int,
        dims: list[int] | None = None
) -> ComplexArray:
    if dims is None:
        dims = [2, 2]
    if circuits_size < 1:
        msg = "integer circuits_size must be larger or equal to 1"
        raise ValueError(msg)

    if target >= circuits_size:
        msg = "target must be integer < integer circuits_size"
        raise ValueError(msg)

    upper = [np.identity(dims[i], dtype="complex") for i in range(target + 1, circuits_size)]
    lower = [np.identity(dims[j], dtype="complex") for j in range(target)]
    circ = [*lower, gate, *upper]
    res = circ[-1]

    for i in reversed(list(range(1, len(circ)))):
        res = np.kron(circ[i - 1], res)

    return res


def apply_gate_to_tlines(
        gate_matrix: ComplexArray,
        circuits_size: int = 2,
        targets: int | list[int] | None = None,
        dims: list[int] | None = None
) -> ComplexArray:
    if dims is None:
        dims = [2, 2]
    if targets is None:
        targets = range(circuits_size)

    if isinstance(targets, int):
        targets = [targets]

    subset_gate = 0
    for i in targets:
        subset_gate += gate_expand_to_circuit(gate_matrix, circuits_size, i, dims)
    return subset_gate

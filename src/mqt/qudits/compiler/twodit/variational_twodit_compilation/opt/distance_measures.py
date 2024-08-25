#!/usr/bin/env python3
"""FROM An alternative quantum fidelity for mixed states of qudits Xiaoguang Wang, 1, 2, * Chang-Shui Yu, 3 and x. x.
Yi 3"""

from __future__ import annotations

import typing

import numpy as np

from mqt.qudits.exceptions.circuiterror import ShapeMismatchError

NDArray = np.ndarray


def size_check(a: NDArray[np.complex128], b: NDArray[np.complex128]) -> bool:
    return bool(a.shape == b.shape and a.shape[0] == a.shape[1])


def fidelity_on_operator(a: np.ndarray, b: np.ndarray) -> float:
    if not size_check(a, b):
        msg = "Input arrays must have the same square shape."
        raise ShapeMismatchError(msg)

    adag = a.T.conj().copy()
    bdag = b.T.conj().copy()
    numerator = np.abs(np.trace(adag @ b))
    denominator = np.sqrt(np.trace(a @ adag) * np.trace(b @ bdag))

    return typing.cast(float, (numerator / denominator))


def fidelity_on_unitares(a: np.ndarray, b: np.ndarray) -> float:
    if not size_check(a, b):
        msg = "Input arrays must have the same square shape."
        raise ShapeMismatchError(msg)

    dimension = a.shape[0]

    return typing.cast(float, np.abs(np.trace(a.T.conj() @ b)) / dimension)


def fidelity_on_density_operator(a: np.ndarray, b: np.ndarray) -> float:
    if not size_check(a, b):
        msg = "Input arrays must have the same square shape."
        raise ShapeMismatchError(msg)

    numerator = np.abs(np.trace(a @ b))
    denominator = np.sqrt(np.trace(a @ a) * np.trace(b @ b))

    return typing.cast(float, (numerator / denominator))


def density_operator(state_vector: list[complex] | np.ndarray[np.complex128]) -> np.ndarray[np.complex128]:
    if isinstance(state_vector, list):
        state_vector = np.array(state_vector)

    return np.outer(state_vector, state_vector.conj())


def frobenius_dist(x: NDArray[np.complex128], y: NDArray[np.complex128]) -> float:
    a = x - y
    return np.sqrt(np.trace(np.abs(a.T.conj() @ a)))


def naive_state_fidelity(state1: NDArray[np.complex128], state2: NDArray[np.complex128]) -> float:
    """
    Calculates fidelity between two state vectors.
    """
    # Ensure both states have the same dimension
    if state1.shape != state2.shape:
        msg = "Input arrays must have the same square shape."
        raise ShapeMismatchError(msg)

    # Inner product of the two states
    inner_product = np.conj(state1).dot(state2)

    # Fidelity calculation
    return np.abs(inner_product) ** 2

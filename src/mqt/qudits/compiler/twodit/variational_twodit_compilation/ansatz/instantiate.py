#!/usr/bin/env python3
from __future__ import annotations

import typing

import numpy as np

from mqt.qudits.compiler.twodit.variational_twodit_compilation.ansatz.ansatz_gen_utils import Primitive
from mqt.qudits.compiler.twodit.variational_twodit_compilation.parametrize import generic_sud, params_splitter
from mqt.qudits.quantum_circuit import gates
from mqt.qudits.quantum_circuit.gates import CustomOne

if typing.TYPE_CHECKING:
    from mqt.qudits.compiler.twodit.variational_twodit_compilation.opt.distance_measures import NDArray
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate


def ansatz_decompose(
    circuit: QuantumCircuit, u: Gate, params: list[list[float] | NDArray[np.float64]], dims: list[int]
) -> list[Gate]:
    counter = 0
    decomposition = []
    for i in range(len(params)):
        if counter == 2:
            counter = 0
            decomposition.append(u)

        decomposition.append(
            CustomOne(circuit, "CUo_SUD", counter, generic_sud(params[i], dims[counter]), dims[counter])
        )

        counter += 1

    return decomposition


def create_cu_instance(circuit: QuantumCircuit, p: list[float] | NDArray[np.float64], dims: list[int]) -> list[Gate]:
    params = params_splitter(p, dims)
    cu = Primitive.CUSTOM_PRIMITIVE
    return ansatz_decompose(circuit, cu, params, dims)


def create_ms_instance(circuit: QuantumCircuit, p: list[float], dims: list[int]) -> list[Gate]:
    params = params_splitter(p, dims)
    ms = gates.MS(circuit, "MS", [0, 1], [np.pi / 2], dims)  # ms_gate(np.pi / 2, dim)

    return ansatz_decompose(circuit, ms, params, dims)


def create_ls_instance(circuit: QuantumCircuit, p: list[float] | NDArray[np.float64], dims: list[int]) -> list[Gate]:
    params = params_splitter(p, dims)

    if 2 in dims:
        theta = np.pi / 2
    elif 3 in dims:
        theta = 2 * np.pi / 3
    else:
        theta = np.pi

    ls = gates.LS(circuit, "LS", [0, 1], [theta], dims)  # ls_gate(theta, dim)

    return ansatz_decompose(circuit, ls, params, dims)

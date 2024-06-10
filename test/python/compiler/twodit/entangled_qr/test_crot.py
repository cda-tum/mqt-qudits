from __future__ import annotations

import operator
from functools import reduce
from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.twodit.entanglement_qr import CRotGen
from mqt.qudits.quantum_circuit import QuantumCircuit


def mini_unitary_sim(circuit, list_of_op):
    size = reduce(operator.mul, circuit.dimensions)
    id_mat = np.identity(size)
    for gate in list_of_op:
        id_mat = gate.to_matrix(identities=2) @ id_mat
        db_mat = id_mat.round(2)
    return id_mat


class TestCRot(TestCase):
    def setUp(self) -> None:
        self.circuit_33 = QuantumCircuit(2, [3, 3], 0)

    def test_crot_101_as_list(self):
        crot_gen = CRotGen(self.circuit_33, [0, 1])
        operations = crot_gen.crot_101_as_list(np.pi / 2, -np.pi / 2)
        crot = mini_unitary_sim(self.circuit_33, operations).round(3)
        x = 0

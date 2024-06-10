from __future__ import annotations

import operator
from functools import reduce
from unittest import TestCase

import numpy as np

from mqt.qudits.compiler.twodit.entanglement_qr import PSwapGen
from mqt.qudits.quantum_circuit import QuantumCircuit
from python.compiler.twodit.entangled_qr.test_crot import mini_unitary_sim


class TestPSwapGen(TestCase):
    def setUp(self) -> None:
        self.circuit_33 = QuantumCircuit(2, [2, 3], 0)

    def test_pswap_101_as_list(self):
        pswap_gen = PSwapGen(self.circuit_33, [0, 1])
        operations = pswap_gen.pswap_101_as_list(np.pi / 3, np.pi / 2)
        pswap = mini_unitary_sim(self.circuit_33, operations).round(3)
        x = 0

from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit
from python.compiler.twodit.entangled_qr.test_entangled_qr import random_unitary_matrix


class TestCustomOne(TestCase):
    @staticmethod
    def test___array__():
        # All 33 csum
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        mru = 1j * np.identity(3) * random_unitary_matrix(3)
        cu = circuit_33.cu_one(0, mru)

        matrix = cu.to_matrix(identities=0)
        assert np.allclose(mru, matrix)

        matrix_dag = cu.dag().to_matrix()
        assert np.allclose(mru.conj().T, matrix_dag)

    @staticmethod
    def test_validate_parameter():
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        cu = circuit_33.cu_one(0, np.identity(3))
        assert cu.validate_parameter(np.identity(3))

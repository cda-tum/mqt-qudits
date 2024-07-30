from unittest import TestCase

import numpy as np

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider


class TestQuditCompiler(TestCase):
    def test_compile_01(self):
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2trits", shots=50)
        qudit_compiler = QuditCompiler()
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        circuit_33.h(0)
        circuit_33.csum([0, 1])
        """
        circuit_33.h(1)
        circuit_33.r(0, [0, 1, np.pi, np.pi / 2])
        circuit_33.r(1, [0, 1, np.pi, np.pi / 2])
        circuit_33.r(1, [0, 1, np.pi / 5, -np.pi / 2]).control([0], [1])
        circuit_33.r(0, [0, 1, np.pi / 3, np.pi / 2])
        """

        circuit = qudit_compiler.compile_O1(backend_ion, circuit_33)
        state = circuit.simulate().round(3)

        x = 0

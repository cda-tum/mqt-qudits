from unittest import TestCase

import numpy as np

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.compiler.compilation_minitools import UnitaryVerifier
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.gates import CustomOne
from mqt.qudits.simulation import MQTQuditProvider
from mqt.qudits.visualisation import plot_state
from python.compiler.twodit.entangled_qr.test_entangled_qr import mini_unitary_sim


class TestQuditCompiler(TestCase):

    def test_compile_00(self):
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2six", shots=50)
        qudit_compiler = QuditCompiler()
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        h = circuit_33.h(0).to_matrix(2).round(2)
        r1 = circuit_33.r(0, [0, 1, np.pi / 7, np.pi / 3]).to_matrix(0).round(2)
        r2 = circuit_33.r(0, [1, 2, np.pi / 5, -np.pi / 3]).to_matrix(0).round(2)
        csum = circuit_33.x(0).dag().to_matrix(2).round(2)

        og_state = circuit_33.simulate().round(5)
        ogp = plot_state(og_state, circuit_33)

        circuit = qudit_compiler.compile_O0(backend_ion, circuit_33)
        state = circuit.simulate().round(5)
        new_s = plot_state(state, circuit)

        assert np.allclose(new_s, ogp)

    def test_compile_01(self):
        pass
        """provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2six", shots=50)
        qudit_compiler = QuditCompiler()
        circuit_33 = QuantumCircuit(2, [3, 3], 0)
        h = circuit_33.h(0).to_matrix(2).round(2)
        r1 = circuit_33.r(0, [0, 1, np.pi / 7, np.pi / 3]).to_matrix(0).round(2)
        r2 = circuit_33.r(0, [1, 2, np.pi / 5, -np.pi / 3]).to_matrix(0).round(2)
        csum = circuit_33.x(0).dag().to_matrix(2).round(2)

        og_state = circuit_33.simulate().round(5)
        ogp = plot_state(og_state, circuit_33)

        circuit = qudit_compiler.compile_O1(backend_ion, circuit_33)
        state = circuit.simulate().round(5)
        new_s = plot_state(state, circuit)

        assert np.allclose(new_s, ogp)"""

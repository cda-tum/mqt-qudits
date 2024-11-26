from __future__ import annotations
from unittest import TestCase

import numpy as np

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_phy_unitary_sim, mini_unitary_sim
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider


class TestPhyMultiSimplePass(TestCase):
    @staticmethod
    def test_multi_transpile_rctrl():
        # Create the original circuit
        circuit = QuantumCircuit(3, [3, 3, 3], 0)
        circuit.r(0, [1, 2, np.pi / 3, np.pi / 7]).control([1, 2], [0, 0])

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps3six")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["PhyMultiSimplePass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        dd = uni_cl.round(3)
        assert np.allclose(uni_l, uni_cl)

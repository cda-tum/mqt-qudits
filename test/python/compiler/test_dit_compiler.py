from __future__ import annotations

import random
from unittest import TestCase

import numpy as np
from numpy.random import choice

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_phy_unitary_sim, mini_unitary_sim, \
    naive_phy_sim
from mqt.qudits.compiler.twodit.variational_twodit_compilation.sparsifier import random_sparse_unitary
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider
from python.compiler.twodit.entangled_qr.test_entangled_qr import random_unitary_matrix


class TestQuditCompiler(TestCase):

    @staticmethod
    def test_compile():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        circuit = QuantumCircuit(4, 2 * [3, 4], 0)
        for i in range(4):
            for j in range(4):
                if choice([True, False]):
                    pass
                    # circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                if choice([True, False]):
                    circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i]*circuit.dimensions[j]))

        insts = random.sample(circuit.instructions, len(circuit.instructions) // 5)
        circuit.set_instructions(insts)
        qudit_compiler = QuditCompiler()
        passes = ["PhyLocQRPass", "PhyEntQRCEXPass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        try:
            assert np.allclose(uni_l, uni_cl)
        except AssertionError:
            pass

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        assert np.allclose(og_state, compiled_state)

    @staticmethod
    def test_transpile():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        circuit = QuantumCircuit(4, 2 * [3, 4], 0)
        for i in range(4):
            for j in range(4):
                if choice([True, False]):
                    circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                if choice([True, False]):
                    circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i]*circuit.dimensions[j]))

        insts = random.sample(circuit.instructions, len(circuit.instructions) // 5)
        circuit.set_instructions(insts)

        qudit_compiler = QuditCompiler()
        passes = ["PhyLocQRPass", "PhyEntSimplePass", "PhyMultiSimplePass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        assert np.allclose(uni_l, uni_cl)

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        assert np.allclose(og_state, compiled_state)

    @staticmethod
    def test_compile_00():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        circuit = QuantumCircuit(4, 2 * [3, 4], 0)
        for i in range(4):
            for j in range(4):
                if choice([True, False]):
                    circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                if choice([True, False]):
                    circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        insts = random.sample(circuit.instructions, len(circuit.instructions) // 5)
        circuit.set_instructions(insts)

        qudit_compiler = QuditCompiler()
        new_circuit = qudit_compiler.compile_O0(backend_ion, circuit)

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        assert np.allclose(uni_l, uni_cl)

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        assert np.allclose(og_state, compiled_state)

    @staticmethod
    def test_compile_O1_resynth():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        circuit = QuantumCircuit(4, 2 * [3, 4], 0)
        for i in range(4):
            for j in range(4):
                if choice([True, False]):
                    circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                if choice([True, False]):
                    circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        insts = random.sample(circuit.instructions, len(circuit.instructions) // 5)
        circuit.set_instructions(insts)

        qudit_compiler = QuditCompiler()
        new_circuit = qudit_compiler.compile_O1_resynth(backend_ion, circuit)

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        assert np.allclose(uni_l, uni_cl)

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        assert np.allclose(og_state, compiled_state)

    @staticmethod
    def test_compile_O1_adaptive():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        circuit = QuantumCircuit(4, 2 * [3, 4], 0)
        for i in range(4):
            for j in range(4):
                if choice([True, False]):
                    circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                if choice([True, False]):
                    circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        insts = random.sample(circuit.instructions, len(circuit.instructions) // 5)
        circuit.set_instructions(insts)

        qudit_compiler = QuditCompiler()
        new_circuit = qudit_compiler.compile_O1_adaptive(backend_ion, circuit)

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        assert np.allclose(uni_l, uni_cl)

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        assert np.allclose(og_state, compiled_state)

    @staticmethod
    def test_compile_02():
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps8seven")

        circuit = QuantumCircuit(4, 2 * [3, 4], 0)
        for i in range(4):
            for j in range(4):
                if choice([True, False]):
                    circuit.cu_one(i, random_sparse_unitary(circuit.dimensions[i]))
                if choice([True, False]):
                    circuit.cu_two([i, j], random_unitary_matrix(circuit.dimensions[i] * circuit.dimensions[j]))

        insts = random.sample(circuit.instructions, len(circuit.instructions) // 5)
        circuit.set_instructions(insts)

        qudit_compiler = QuditCompiler()
        new_circuit = qudit_compiler.compile_O2(backend_ion, circuit)

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_phy_unitary_sim(new_circuit)
        assert np.allclose(uni_l, uni_cl)

        og_state = circuit.simulate()
        compiled_state = naive_phy_sim(new_circuit)
        assert np.allclose(og_state, compiled_state)

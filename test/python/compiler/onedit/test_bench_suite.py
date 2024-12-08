from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_unitary_sim
from mqt.qudits.compiler.onedit.randomized_benchmarking.bench_suite import (
    generate_clifford_group,
    get_h_gate,
    get_package_data_path,
    get_s_gate,
    load_clifford_group_from_file,
    matrix_hash,
    save_clifford_group_to_file,
)
from mqt.qudits.quantum_circuit import QuantumCircuit, QuantumRegister
from mqt.qudits.simulation import MQTQuditProvider


class TestCliffordGroupGeneration(unittest.TestCase):
    def test_get_h_gate(self):
        h_gate = get_h_gate(2)
        expected_h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        np.testing.assert_array_almost_equal(h_gate, expected_h)

    def test_get_s_gate(self):
        s_gate = get_s_gate(2)
        expected_s = np.array([[1, 0], [0, 1j]])
        np.testing.assert_array_almost_equal(s_gate, expected_s)

    def test_matrix_hash(self):
        matrix1 = np.array([[1, 0], [0, 1]])
        matrix2 = np.array([[1, 0], [0, 1]])
        matrix3 = np.array([[0, 1], [1, 0]])

        assert matrix_hash(matrix1) == matrix_hash(matrix2)
        assert matrix_hash(matrix1) != matrix_hash(matrix3)

    def test_generate_clifford_group(self):
        clifford_group = generate_clifford_group(2, max_length=2)

        # Check if the identity matrix is in the group
        assert np.allclose(clifford_group["h"], get_h_gate(2))
        assert np.allclose(clifford_group["s"], get_s_gate(2))

        # Check if the group has the expected number of elements
        # For qubit (d=2) and max_length=2, we expect 6 unique elements
        assert len(clifford_group) == 6

    def test_get_package_data_path(self):
        filename = "test_file.pkl"
        path = get_package_data_path(filename)
        assert os.path.dirname(path).endswith("data")
        assert os.path.basename(path) == filename

    def test_save_and_load_clifford_group(self):
        clifford_group = generate_clifford_group(2, max_length=2)

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            filename = os.path.basename(temp_file.name)

        try:
            save_clifford_group_to_file(clifford_group, filename)
            loaded_group = load_clifford_group_from_file(filename)

            assert len(clifford_group) == len(loaded_group)
            for key in clifford_group:
                np.testing.assert_array_almost_equal(clifford_group[key], loaded_group[key])
        finally:
            os.unlink(get_package_data_path(filename))

    def test_benching(self):
        DIM = 3

        clifford_group = generate_clifford_group(DIM, max_length=10)  # What max length?
        save_clifford_group_to_file(clifford_group, f"cliffords_{DIM}.dat")

        DIM = 3
        clifford_group = load_clifford_group_from_file(f"cliffords_{DIM}.dat")

        def create_rb_sequence(length=2):
            circuit = QuantumCircuit()

            dit_register = QuantumRegister("dits", 1, [3])

            circuit.append(dit_register)

            inversion = np.eye(DIM)

            check = np.eye(DIM)

            for _ in range(length):
                random_gate = list(clifford_group.values())[np.random.randint(len(clifford_group))]
                inversion = inversion @ random_gate.conj().T
                circuit.cu_one(0, random_gate)
                check = random_gate @ check
            circuit.cu_one(0, inversion)

            # print(f"Number of operations: {len(circuit.instructions)}")
            # print(f"Number of qudits in the circuit: {circuit.num_qudits}")
            return circuit

        circuit = create_rb_sequence(length=16)
        compiled_cirucit = circuit.compileO1("faketraps2trits", "adapt")

        uni = mini_unitary_sim(compiled_cirucit).round(4)
        v = np.eye(DIM)[:, compiled_cirucit.final_mappings[0]]
        v2 = np.eye(DIM)
        tpuni = uni @ v
        tpuni = v2.T @ tpuni  # Pi dag

        # print(f"Number of operations: {len(compiled_cirucit.instructions)}")

        provider = MQTQuditProvider()
        backend = provider.get_backend("faketraps2trits")

        # print(backend.energy_level_graphs[0].edges)
        # print(compiled_cirucit.final_mappings[0])

        # for instruction in compiled_cirucit.instructions:
        #    print(instruction.lev_a, instruction.lev_b)
        #    print(instruction.theta, instruction.phi)
        # circuit.simulate()

        # very_crude_backend(compiled_cirucit, "localhost:5173")
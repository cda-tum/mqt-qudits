from __future__ import annotations

from collections import defaultdict
from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.simulation.noise_tools import NoiseModel, NoisyCircuitFactory, SubspaceNoise


def rand_0_5() -> int:
    rng = np.random.default_rng()
    return int(rng.integers(0, 6))


class TestNoisyCircuitFactoryPhysical(TestCase):
    def setUp(self) -> None:
        sub1 = SubspaceNoise(0.999, 0.999, (0, 1))
        sub2 = SubspaceNoise(0.0, 0.999, [(1, 2), (2, 3)])
        sub3 = SubspaceNoise(0.999, 0.0, [(1, 2), (2, 3)])

        # Add errors to noise_tools model

        noise_model = NoiseModel()  # We know that the architecture is only two qudits
        # Very noisy gate_matrix
        noise_model.add_all_qudit_quantum_error(sub1, ["csum"])
        # Local Gates
        noise_model.add_quantum_error_locally(sub1, ["z"])
        noise_model.add_quantum_error_locally(sub2, ["rh", "h", "rxy", "x"])
        noise_model.add_quantum_error_locally(sub3, ["s"])
        self.noise_model = noise_model

    def test_generate_circuit(self):
        qreg_example = QuantumRegister("reg", 6, 6 * [5])
        circ = QuantumCircuit(qreg_example)
        choice = rand_0_5()
        x = circ.x(choice)
        x.control([int(np.mod(choice + 1, 5))], [2])
        circ.rz(rand_0_5(), [0, 2, np.pi / 13])
        circ.cx([3, 4], [0, 3, 0, np.pi / 12])
        circ.x(rand_0_5()).dag()
        circ.s(rand_0_5())
        circ.z(rand_0_5())
        circ.csum([5, 1])
        circ.virtrz(rand_0_5(), [0, np.pi / 13]).dag()
        circ.virtrz(rand_0_5(), [1, -np.pi / 8])
        circ.csum([2, 5]).dag()
        circ.x(rand_0_5()).dag()
        circ.z(rand_0_5()).dag()
        circ.h(rand_0_5())
        circ.rz(rand_0_5(), [3, 4, np.pi / 13]).dag()
        circ.h(rand_0_5()).dag()
        circ.r(rand_0_5(), [0, 1, np.pi / 5 + np.pi, np.pi / 7])
        circ.rh(rand_0_5(), [1, 3])
        circ.r(rand_0_5(), [0, 4, np.pi, np.pi / 2]).dag()
        circ.r(rand_0_5(), [0, 3, np.pi / 5, np.pi / 7])
        circ.cx([1, 2], [0, 1, 1, np.pi / 2]).dag()
        circ.csum([0, 1])

        factory = NoisyCircuitFactory(self.noise_model, circ)
        instructions_og = circ.instructions
        new_circ = factory.generate_circuit()
        instructions_new = new_circ.instructions

        tag_counts_list1: dict[str, int] = defaultdict(int)
        tag_counts_list2: dict[str, int] = defaultdict(int)
        insts = 0
        insts_new = 0
        for gate in instructions_og:
            insts += 1
            tag_counts_list1[gate.qasm_tag] += 1

        for gate in instructions_new:
            insts_new += 1
            tag_counts_list2[gate.qasm_tag] += 1

        keys_to_check = ["noisex", "virtrz"]
        valid_stochasticity = True
        # Iterate over all keys
        for key in tag_counts_list1.keys() | tag_counts_list2.keys():
            if key in keys_to_check:
                if tag_counts_list1.get(key, 0) > tag_counts_list2.get(key, 0):
                    valid_stochasticity = False
                    print(key, "error")
            elif tag_counts_list1.get(key, 0) != tag_counts_list2.get(key, 0):
                valid_stochasticity = False
                print(key, "error")

        assert valid_stochasticity
        assert insts == circ.number_gates
        assert insts_new == new_circ.number_gates
        assert len(instructions_new) > len(instructions_og)

    def test_generate_circuit_isolated1(self):
        qreg_example = QuantumRegister("reg", 2, [5, 5])
        circ = QuantumCircuit(qreg_example)
        circ.z(0)
        circ.z(1)

        factory = NoisyCircuitFactory(self.noise_model, circ)
        instructions_og = circ.instructions
        new_circ = factory.generate_circuit()
        assert circ.number_gates == 2  # original x only
        assert [i.qasm_tag for i in instructions_og] == ["z", "z"]
        for tag in [i.qasm_tag for i in new_circ.instructions]:
            assert tag in {"z", "virtrz", "noisex"}

    def test_generate_circuit_isolated2(self):
        qreg_example = QuantumRegister("reg", 2, [5, 5])
        circ = QuantumCircuit(qreg_example)
        circ.x(0)
        circ.x(1)

        factory = NoisyCircuitFactory(self.noise_model, circ)
        instructions_og = circ.instructions
        new_circ = factory.generate_circuit()
        assert circ.number_gates == 2  # original x only
        assert [i.qasm_tag for i in instructions_og] == ["x", "x"]
        for tag in [i.qasm_tag for i in new_circ.instructions]:
            assert tag in {"x", "virtrz"}

    def test_generate_circuit_isolated3(self):
        qreg_example = QuantumRegister("reg", 2, [5, 5])
        circ = QuantumCircuit(qreg_example)
        circ.s(0)
        circ.s(1)

        factory = NoisyCircuitFactory(self.noise_model, circ)
        instructions_og = circ.instructions
        new_circ = factory.generate_circuit()
        assert circ.number_gates == 2  # original x only
        assert [i.qasm_tag for i in instructions_og] == ["s", "s"]
        for tag in [i.qasm_tag for i in new_circ.instructions]:
            assert tag in {"s", "noisex", "virtrz"}

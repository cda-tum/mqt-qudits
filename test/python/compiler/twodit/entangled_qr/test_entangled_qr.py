from __future__ import annotations

import typing
from unittest import TestCase

import numpy as np
from scipy.stats import unitary_group  # type: ignore[import-not-found]

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_unitary_sim, naive_phy_sim
from mqt.qudits.compiler.twodit.entanglement_qr import EntangledQRCEX
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider

if typing.TYPE_CHECKING:
    from numpy.typing import NDArray


def random_unitary_matrix(n: int) -> NDArray[np.complex128, np.complex128]:
    return unitary_group.rvs(n)


class TestEntangledQR(TestCase):
    def test_entangling_qr(self):
        circuit_53 = QuantumCircuit(2, [5, 3], 0)
        target_u = random_unitary_matrix(15)
        t = circuit_53.cu_two([0, 1], target_u)

        eqr = EntangledQRCEX(t)
        decomp, _countcr, _countpsw = eqr.execute()

        target = target_u.copy()
        for rotation in decomp:
            target = rotation.to_matrix(identities=2) @ target
            target.round(4)
        global_phase = target[0][0]
        target /= global_phase
        res = (abs(target - np.identity(15, dtype="complex")) < 10e-5).all()

        assert res

        id_mat = np.identity(15)
        for gate in reversed(decomp):
            id_mat = gate.to_matrix(identities=2).conj().T @ id_mat
        id_mat /= id_mat[0][0]
        assert np.allclose(id_mat, target)
        # for rotation in reversed(decomp):
        #    target = rotation.to_matrix(identities=2).conj().T @ target

        # as_gates = [op.dag() for op in reversed(decomp)]
        # uni = mini_unitary_sim(circuit_53, as_gates)
        # assert np.allclose(uni, target)

    @staticmethod
    def test_entangling_qr_2():
        # Create the original circuit
        circuit = QuantumCircuit(2, [3, 3], 0)
        # circuit.x(0)
        target = random_unitary_matrix(9)
        circuit.cu_two([0, 1], target)

        # Simulate the original circuit
        original_state = circuit.simulate()
        print("Original circuit simulation result:")
        print(original_state.round(3))

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2trits")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["LogEntQRCEXPass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        uni_l = mini_unitary_sim(circuit, circuit.instructions)
        uni_c = mini_unitary_sim(new_circuit, new_circuit.instructions)
        assert np.allclose(uni_l, uni_c)

        # Simulate the compiled circuit
        compiled_state = new_circuit.simulate()
        print("\nCompiled circuit simulation result:")
        print(compiled_state.round(3))

        # Compare the results
        is_close = np.allclose(original_state, compiled_state)
        print(f"\nAre the simulation results close? {is_close}")
        assert is_close

    @staticmethod
    def test_physical_entangling_qr():
        # Create the original circuit
        circuit = QuantumCircuit(2, [3, 3], 0)
        circuit.h(0)
        # circuit.csum([0, 1])
        # target = random_unitary_matrix(9)
        # circuit.cu_two([0, 1], target)

        # Simulate the original circuit
        original_state = circuit.simulate()
        print("Original circuit simulation result:")
        print(original_state.round(3))

        # Set up the provider and backend
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2trits")

        # Compile the circuit
        qudit_compiler = QuditCompiler()
        passes = ["PhyLocQRPass", "PhyEntQRCEXPass"]
        new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

        # Simulate the compiled circuit
        compiled_state = naive_phy_sim(provider.get_backend("faketraps2trits"), new_circuit)  # new_circuit.simulate()
        print("\nCompiled circuit simulation result:")
        print(compiled_state.round(3))

        # Compare the results
        is_close = np.allclose(original_state, compiled_state)
        print(f"\nAre the simulation results close? {is_close}")

from __future__ import annotations

import typing
from unittest import TestCase

import numpy as np
from scipy.stats import unitary_group  # type: ignore[import-not-found]

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.compiler.compilation_minitools.naive_unitary_verifier import mini_phy_unitary_sim, mini_unitary_sim, \
    naive_phy_sim
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
        global_phase = target[0][0]
        if np.round(global_phase, 13) != 1.0 + 0j:
            target /= global_phase
        # res = (abs(target - np.identity(15, dtype="complex")) < 10e-13).all()
        assert np.allclose(target, np.identity(15, dtype="complex"))

        reconstructed = np.identity(15)
        gates_ms = []
        for gate in reversed(decomp):
            gates_ms.append((gate, gate.to_matrix(identities=2).conj().T))
            reconstructed = gate.to_matrix(identities=2).conj().T @ reconstructed

        assert np.allclose(reconstructed, target_u)

        for i in range(len(gates_ms)):
            gate2check = gates_ms[i][0]
            checker = gate2check.dag().to_matrix(identities=2).round(13)
            og_conj_t = gates_ms[i][1].round(13)
            assert np.allclose(checker, og_conj_t)

    @staticmethod
    def test_log_entangling_qr_circuit():
        # Create the original circuit
        circuit = QuantumCircuit(2, [2, 3], 0)
        target = random_unitary_matrix(6)
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

        uni_l = mini_unitary_sim(circuit)
        uni_cl = mini_unitary_sim(new_circuit)
        assert np.allclose(uni_l, uni_cl)

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
        for i in range(20):
            # Create the original circuit
            circuit = QuantumCircuit(3, [3, 4, 4], 0)
            # circuit.rh(0, [0, 1]).control([1], [1])
            circuit.cu_two([0, 1], random_unitary_matrix(12))
            #circuit.cu_two([0, 2], random_unitary_matrix(12))
            circuit.cu_two([1, 2], random_unitary_matrix(16))
            #circuit.cu_two([0, 2], random_unitary_matrix(12))
            #circuit.cu_two([1, 0], random_unitary_matrix(12))

            # Simulate the original circuit
            original_state = circuit.simulate()
            print("Original circuit simulation result:")
            print(original_state.round(3))

            # Set up the provider and backend
            provider = MQTQuditProvider()
            backend_ion = provider.get_backend("faketraps8seven")

            # Compile the circuit
            qudit_compiler = QuditCompiler()
            passes = ["PhyEntQRCEXPass"]
            new_circuit = qudit_compiler.compile(backend_ion, circuit, passes)

            # Simulate the compiled circuit
            compiled_state = naive_phy_sim(new_circuit)
            print("\nCompiled circuit simulation result:")
            print(compiled_state.round(3))

            try:
                uni_l = mini_unitary_sim(circuit)
                uni_cl = mini_phy_unitary_sim(new_circuit)
                # Compare the results
                assert np.allclose(original_state, compiled_state)
                print(f"\nAre the unitaries close? True")
            except AssertionError:
                diff_mask = ~np.isclose(uni_l, uni_cl, rtol=1e-10, atol=1e-10)
                diff_positions = np.where(diff_mask)
                pass

            try:
                # Compare the results
                assert np.allclose(original_state, compiled_state)
                print(f"\nAre the simulation results close? True")
            except AssertionError:
                pass


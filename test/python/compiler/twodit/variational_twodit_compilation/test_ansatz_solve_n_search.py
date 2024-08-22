from __future__ import annotations

from unittest import TestCase

from mqt.qudits.compiler.twodit.variational_twodit_compilation.layered_compilation import variational_compile
from mqt.qudits.compiler.twodit.variational_twodit_compilation.opt import fidelity_on_unitares
from mqt.qudits.quantum_circuit import QuantumCircuit
from python.compiler.twodit.entangled_qr.test_entangled_qr import mini_unitary_sim


class TestAnsatzSearch(TestCase):
    def test_binary_search_compile(self) -> None:
        self.circuit_og = QuantumCircuit(2, [2, 2], 0)
        cx = self.circuit_og.cx([0, 1])
        circuit = variational_compile(cx, 1e-2, "MS", 1)
        op = mini_unitary_sim(circuit, circuit.instructions)
        f = fidelity_on_unitares(op, cx.to_matrix())
        assert 0 < f < 1

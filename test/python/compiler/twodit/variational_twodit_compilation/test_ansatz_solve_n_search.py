from unittest import TestCase

from mqt.qudits.compiler.twodit.variational_twodit_compilation.layered_compilation import variational_compile
from mqt.qudits.quantum_circuit import QuantumCircuit
from python.compiler.twodit.entangled_qr.test_entangled_qr import mini_unitary_sim


class TestAnsatzSearch(TestCase):
    def test_binary_search_compile(self) -> None:
        self.circuit = QuantumCircuit(2, [2, 3], 0)
        cx = self.circuit.cx([0, 1])
        circuit = variational_compile(cx, 1e-1, "MS", 1)
        op = mini_unitary_sim(self.circuit, circuit.instructions).round(3)
        print("done")

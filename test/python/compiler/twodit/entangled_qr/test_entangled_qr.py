from __future__ import annotations

import operator
from functools import reduce
from unittest import TestCase

import numpy as np
from scipy.stats import unitary_group

from mqt.qudits.compiler import QuditCompiler
from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.simulation import MQTQuditProvider


def mini_unitary_sim(circuit, list_of_op):
    size = reduce(operator.mul, circuit.dimensions)
    id_mat = np.identity(size)
    for gate in list_of_op:
        id_mat = gate.to_matrix(identities=2) @ id_mat
        db_mat = id_mat.round(2)
    return id_mat


def random_unitary_matrix(n):
    return unitary_group.rvs(n)


class TestEntangledQR(TestCase):
    def setUp(self) -> None:
        MQTQuditProvider()

        self.circuit_33 = QuantumCircuit(2, [5, 3], 0)
        self.circuit_s = QuantumCircuit(2, [5, 3], 0)
        #self.u = self.circuit_33.r(1, [0, 1, np.pi / 3, np.pi / 5]).control([0], [2])

    def test_entangling_qr(self):
        #pswap_gen = PSwapGen(self.circuit_33, [0, 1])
        #p_op = pswap_gen.permute_pswap_101_as_list(2, np.pi / 4, -np.pi / 3)
        #p_op_z = CZRotGen(self.circuit_33, [0, 1]).z_pswap_101_as_list(5, np.pi/3)
        #moved_crot = mini_unitary_sim(self.circuit_33, p_op_z).round(3)
        target = random_unitary_matrix(15)
        # r = self.circuit_s.r(1, [0, 1, 1.0471975511965972, -2.513274122871836]).control([0], [2]).to_matrix()
        # check = (r @ matrix).round()
        self.circuit_33.cu_two([0, 1], target)
        provider = MQTQuditProvider()
        backend_ion = provider.get_backend("faketraps2trits", shots=1000)
        qudit_compiler = QuditCompiler()

        passes = ["LogEntQRCEXPass"]
        new_circuit = qudit_compiler.compile(backend_ion, self.circuit_33, passes)

        for rotation in new_circuit.instructions:
            target = rotation.to_matrix(identities=2) @ target
        target = target / target[0][0]
        res = (abs(target - np.identity(15, dtype='complex')) < 10e-5).all()

        compiled = mini_unitary_sim(self.circuit_33, new_circuit.instructions).round(3)
        self.assertTrue(np.allclose(compiled, target))

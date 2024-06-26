from __future__ import annotations

from unittest import TestCase

import numpy as np

from mqt.qudits.quantum_circuit import QuantumCircuit


class TestGellMann(TestCase):
    def test___array__(self):
        circuit = QuantumCircuit(1, [4], 0)

        Ga = circuit.gellmann(0, [0, 1, "a"])
        Gs = circuit.gellmann(0, [0, 1, "s"])
        # Ga = GellMann(0, 1, 'a', dimension)
        # Gs = GellMann(0, 1, 's', dimension)

        test_Ga = np.array([
            [0.0 + 0.0j, 0.0 - 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ])

        test_Gs = np.array([
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ])

        assert np.allclose(Ga.to_matrix(identities=0), test_Ga)
        assert np.allclose(Gs.to_matrix(identities=0), test_Gs)

        circuit = QuantumCircuit(1, [3], 0)
        Ga = circuit.gellmann(0, [0, 1, "a"])
        Gs = circuit.gellmann(0, [0, 1, "s"])
        # Ga = GellMann(0, 1, 'a', dimension)
        # Gs = GellMann(0, 1, 's', dimension)

        test_Ga = np.array([
            [0.0 + 0.0j, 0.0 - 1.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ])

        test_Gs = np.array([
            [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ])

        assert np.allclose(Ga, test_Ga)
        assert np.allclose(Gs, test_Gs)

        circuit = QuantumCircuit(1, [3], 0)
        Ga = circuit.gellmann(0, [0, 2, "a"])
        Gs = circuit.gellmann(0, [0, 2, "s"])
        # Ga = GellMann(0, 2, 'a', dimension)
        # Gs = GellMann(0, 2, 's', dimension)

        test_Ga = np.array([
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 - 1.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ])

        test_Gs = np.array([
            [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
        ])

        assert np.allclose(Ga, test_Ga)
        assert np.allclose(Gs, test_Gs)

    def test_validate_parameter(self):
        circuit = QuantumCircuit(1, [3], 0)
        Ga = circuit.gellmann(0, [0, 2, "a"])
        Gs = circuit.gellmann(0, [0, 2, "s"])
        assert Ga.validate_parameter([0, 2, "a"])
        assert Gs.validate_parameter([0, 2, "s"])

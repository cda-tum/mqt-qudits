from __future__ import annotations

from typing import TYPE_CHECKING

from ..quantum_circuit import QuantumCircuit, gates
from ..quantum_circuit.gate import GateTypes

if TYPE_CHECKING:
    from collections.abc import Callable


def draw_qudit_local(circuit: QuantumCircuit) -> None:
    gate_representations: dict[type, Callable] = {
        gates.VirtRz: lambda g: f"--[VRz{g.lev_a}({g.phi:.2f})]--",
        gates.R: lambda g: f"--[R{g.lev_a}{g.lev_b}({g.theta:.2f},{g.phi:.2f})]--",
        gates.CustomOne: lambda _: "--[CuOne]--",
    }

    for line in range(circuit.num_qudits):
        print("|0>---", end="")
        for gate in circuit.instructions:
            if gate.gate_type == GateTypes.SINGLE and line == gate.target_qudits:
                gate_repr = gate_representations.get(type(gate), lambda _: "--G--")
                print(gate_repr(gate), end="")
            else:
                print("--MG--", end="")
        print("---=||")

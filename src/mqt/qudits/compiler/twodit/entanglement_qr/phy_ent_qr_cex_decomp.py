from __future__ import annotations

import gc
import typing
from typing import List, cast

from mqt.qudits.compiler import CompilerPass
from mqt.qudits.compiler.onedit import PhyLocAdaPass
from mqt.qudits.compiler.twodit.entanglement_qr import EntangledQRCEX
from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes
from mqt.qudits.quantum_circuit.gates import Perm

if typing.TYPE_CHECKING:
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate
    from mqt.qudits.simulation.backends.backendv2 import Backend


class PhyEntQRCEXPass(CompilerPass):
    def __init__(self, backend: Backend) -> None:
        super().__init__(backend)
        from mqt.qudits.quantum_circuit import QuantumCircuit

        self.circuit = QuantumCircuit()

    def transpile_gate(self, gate: Gate) -> list[Gate]:
        target_qudits = cast(List[int], gate.target_qudits)
        dimensions = cast(List[int], gate.dimensions)

        energy_graph_c = self.backend.energy_level_graphs[target_qudits[0]]
        energy_graph_t = self.backend.energy_level_graphs[target_qudits[1]]

        lp_map_0 = [lev for lev in energy_graph_c.log_phy_map if lev < dimensions[target_qudits[0]]]
        lp_map_1 = [lev for lev in energy_graph_t.log_phy_map if lev < dimensions[target_qudits[1]]]

        perm_0 = Perm(gate.parent_circuit, "Pm_ent_0", target_qudits[0], lp_map_0, dimensions[0])
        perm_1 = Perm(gate.parent_circuit, "Pm_ent_1", target_qudits[1], lp_map_1, dimensions[1])
        perm_0_dag = Perm(gate.parent_circuit, "Pm_ent_0", target_qudits[0], lp_map_0, dimensions[0]).dag()
        perm_1_dag = Perm(gate.parent_circuit, "Pm_ent_1", target_qudits[1], lp_map_1, dimensions[1]).dag()

        phyloc = PhyLocAdaPass(self.backend)
        perm_0_seq = phyloc.transpile_gate(perm_0)
        perm_1_seq = phyloc.transpile_gate(perm_1)
        perm_0_d_seq = phyloc.transpile_gate(perm_0_dag)
        perm_1_d_seq = phyloc.transpile_gate(perm_1_dag)

        eqr = EntangledQRCEX(gate)
        decomp, _countcr, _countpsw = eqr.execute()
        perm_0_d_seq.extend(perm_1_d_seq)
        perm_0_d_seq.extend(decomp)
        perm_0_d_seq.extend(perm_0_seq)
        perm_0_d_seq.extend(perm_1_seq)

        return [op.dag() for op in reversed(decomp)]

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions = []

        for gate in instructions:
            if gate.gate_type == GateTypes.TWO:
                gate_trans = self.transpile_gate(gate)
                new_instructions.extend(gate_trans)
                gc.collect()
            else:
                new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)

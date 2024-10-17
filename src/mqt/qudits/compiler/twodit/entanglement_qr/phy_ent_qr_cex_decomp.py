from __future__ import annotations

import gc
from typing import TYPE_CHECKING, cast

from mqt.qudits.compiler import CompilerPass
from mqt.qudits.compiler.onedit import PhyLocQRPass
from mqt.qudits.compiler.twodit.entanglement_qr import EntangledQRCEX
from mqt.qudits.core.custom_python_utils import append_to_front
from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes
from mqt.qudits.quantum_circuit.gates import CEx, Perm

if TYPE_CHECKING:
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate
    from mqt.qudits.simulation.backends.backendv2 import Backend


class PhyEntQRCEXPass(CompilerPass):
    def __init__(self, backend: Backend) -> None:
        super().__init__(backend)
        from mqt.qudits.quantum_circuit import QuantumCircuit

        self.circuit = QuantumCircuit()

    def __transpile_local_ops(self, gate: Gate):
        phyloc = PhyLocQRPass(self.backend)
        return phyloc.transpile_gate(gate)

    def transpile_gate(self, gate: Gate) -> list[Gate]:
        def check_lev(lev, dim):
            if lev < dim:
                return lev
            msg = "Mapping Not Compatible with Circuit."
            raise IndexError(msg)

        target_qudits = cast(list[int], gate.target_qudits)
        dimensions = cast(list[int], gate.dimensions)

        energy_graph_c = self.backend.energy_level_graphs[target_qudits[0]]
        energy_graph_t = self.backend.energy_level_graphs[target_qudits[1]]

        lp_map_0 = [check_lev(lev, dimensions[0]) for lev in energy_graph_c.log_phy_map]
        lp_map_1 = [check_lev(lev, dimensions[1]) for lev in energy_graph_t.log_phy_map]

        if isinstance(gate, CEx):
            parent_circ = gate.parent_circuit
            new_ctrl_lev = lp_map_0[gate.ctrl_lev]
            new_la = lp_map_1[gate.lev_a]
            new_lb = lp_map_1[gate.lev_b]
            if new_la < new_lb:
                new_parameters = [new_la, new_lb, new_ctrl_lev, gate.phi]
            else:
                new_parameters = [new_lb, new_la, new_ctrl_lev, gate.phi]
            tcex = CEx(parent_circ, "CEx_t" + str(target_qudits), target_qudits, new_parameters, dimensions, None)
            return [tcex]

        perm_0 = Perm(gate.parent_circuit, "Pm_ent_0", target_qudits[0], lp_map_0, dimensions[0])
        perm_1 = Perm(gate.parent_circuit, "Pm_ent_1", target_qudits[1], lp_map_1, dimensions[1])
        perm_0_dag = Perm(gate.parent_circuit, "Pm_ent_0", target_qudits[0], lp_map_0, dimensions[0]).dag()
        perm_1_dag = Perm(gate.parent_circuit, "Pm_ent_1", target_qudits[1], lp_map_1, dimensions[1]).dag()

        eqr = EntangledQRCEX(gate)
        decomp, _countcr, _countpsw = eqr.execute()

        # seq_perm_0_d = self.__transpile_local_ops(perm_0_dag)
        # seq_perm_1_d = self.__transpile_local_ops(perm_1_dag)
        # seq_perm_0 = self.__transpile_local_ops(perm_0)
        # seq_perm_1 = self.__transpile_local_ops(perm_1)

        full_sequence = [perm_0_dag, perm_1_dag]
        full_sequence.extend(decomp)
        full_sequence.extend((perm_0, perm_1))

        physical_sequence = []
        for gate in reversed(decomp):
            if gate.gate_type == GateTypes.SINGLE:
                loc_gate = self.__transpile_local_ops(gate)
                physical_sequence.extend(loc_gate)
            else:
                physical_sequence.append(gate)

        [op.dag() for op in reversed(physical_sequence)]

        # full_sequence.extend(seq_perm_0_d)
        # full_sequence.extend(seq_perm_1_d)
        # full_sequence.extend(physical_sequence_dag)
        # full_sequence.extend(seq_perm_0)
        # full_sequence.extend(seq_perm_1)

        return full_sequence

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions = []

        for gate in reversed(instructions):
            if gate.gate_type == GateTypes.TWO:
                gate_trans = self.transpile_gate(gate)
                append_to_front(new_instructions, gate_trans)
                # new_instructions.extend(gate_trans)
                gc.collect()
            else:
                append_to_front(new_instructions, gate)
                # new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)

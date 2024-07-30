import gc

from mqt.qudits.compiler import CompilerPass
from mqt.qudits.compiler.twodit.entanglement_qr import EntangledQRCEX
from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes
from mqt.qudits.quantum_circuit.gates import Perm


class PhyEntQRCEXPass(CompilerPass):
    def __init__(self, backend) -> None:
        super().__init__(backend)
        self.circuit = None

    def traspile_gate(self, gate):
        energy_graph_c = self.backend.energy_level_graphs[gate._target_qudits[0]]
        energy_graph_t = self.backend.energy_level_graphs[gate._target_qudits[1]]
        lp_map_0 = energy_graph_c.log_phy_map
        lp_map_1 = energy_graph_t.log_phy_map
        perm_0 = Perm(gate.parent_circuit, "Pm_ent_0", gate._target_qudits[0], lp_map_0, gate._dimensions[0])
        perm_1 = Perm(gate.parent_circuit, "Pm_ent_1", gate._target_qudits[1], lp_map_1, gate._dimensions[1])
        perm_0_dag = Perm(gate.parent_circuit, "Pm_ent_0", gate._target_qudits[0], lp_map_0, gate._dimensions[0]).dag()
        perm_1_dag = Perm(gate.parent_circuit, "Pm_ent_1", gate._target_qudits[1], lp_map_1, gate._dimensions[1]).dag()

        eqr = EntangledQRCEX(gate)
        decomp, countcr, countpsw = eqr.execute()
        decomp.insert(0, perm_0)
        decomp.insert(0, perm_1)
        decomp.append(perm_0_dag)
        decomp.append(perm_1_dag)
        return decomp

    def transpile(self, circuit):
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions = []

        for gate in instructions:
            if gate.gate_type == GateTypes.TWO:
                decomp = self.traspile_gate(gate)
                new_instructions += decomp
                gc.collect()
            else:
                new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)

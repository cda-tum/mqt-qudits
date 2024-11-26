from __future__ import annotations

import gc
from typing import TYPE_CHECKING, cast

from mqt.qudits.compiler import CompilerPass
from mqt.qudits.compiler.compilation_minitools.local_compilation_minitools import check_lev
from mqt.qudits.core.custom_python_utils import append_to_front
from mqt.qudits.quantum_circuit.components.extensions.controls import ControlData
from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes


if TYPE_CHECKING:
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate
    from mqt.qudits.simulation.backends.backendv2 import Backend
    from mqt.qudits.core import LevelGraph

class PhyMultiSimplePass(CompilerPass):
    def __init__(self, backend: Backend) -> None:
        super().__init__(backend)
        from mqt.qudits.quantum_circuit import QuantumCircuit
        self.circuit = QuantumCircuit()

    def __routing(self, gate: R, graph: LevelGraph):
        from mqt.qudits.compiler.onedit.local_operation_swap import cost_calculator
        from mqt.qudits.compiler.onedit.local_operation_swap import gate_chain_condition
        from mqt.qudits.quantum_circuit.gates import R
        phi = gate.phi
        _, pi_pulses_routing, temp_placement, _, _ = cost_calculator(gate, graph, 0)

        if temp_placement.nodes[gate.lev_a]["lpmap"] > temp_placement.nodes[gate.lev_b]["lpmap"]:
            phi *= -1

        physical_rotation = R(
                self.circuit,
                "R",
                gate.target_qudits,
                [temp_placement.nodes[gate.lev_a]["lpmap"],
                 temp_placement.nodes[gate.lev_b]["lpmap"], gate.theta, phi],
                gate.dimensions
        )

        physical_rotation = gate_chain_condition(pi_pulses_routing, physical_rotation)
        pi_backs = []

        for pi_g in reversed(pi_pulses_routing):
            pi_backs.append(
                    R(
                            self.circuit,
                            "R",
                            gate.target_qudits,
                            [pi_g.lev_a, pi_g.lev_b, pi_g.theta, -pi_g.phi],
                            gate.dimensions
                    )
            )
        return pi_pulses_routing, physical_rotation, pi_backs

    def transpile_gate(self, gate: Gate) -> list[Gate]:
        from mqt.qudits.quantum_circuit.gates import R
        assert gate.gate_type == GateTypes.MULTI
        self.circuit = gate.parent_circuit

        if isinstance(gate.target_qudits, int):
            gate_controls = gate.control_info["controls"]
            indices = gate_controls.indices
            states = gate_controls.ctrl_states
            target_qudits = indices + [gate.target_qudits]
            dimensions = [gate.parent_circuit.dimensions[i] for i in target_qudits]
        else:
            target_qudits = cast(list[int], gate.target_qudits)
            dimensions = cast(list[int], gate.dimensions)

        # Get energy graphs for all control qudits and target qudit
        energy_graphs = {
            qudit: self.backend.energy_level_graphs[qudit]
            for qudit in target_qudits
        }

        # Create logical-to-physical mapping for all qudits
        lp_maps = {
            qudit: [check_lev(lev, dim)
                    for lev in energy_graphs[qudit].log_phy_map[:dim]]
            for qudit, dim in zip(target_qudits, dimensions)
        }

        if isinstance(gate, R):
            if len(indices) > 0:
                assert len(states) > 0 and len(states) == len(indices)

                # Create ghost rotation for routing
                target_qudit = target_qudits[-1]  # Last qudit is the target
                ghost_rotation = R(self.circuit,
                                   f"R_ghost_t{target_qudit}",
                                   target_qudit,
                                   [gate.lev_a, gate.lev_b, gate.theta, gate.phi],
                                   dimensions[-1],
                                   None)

                # Get routing operations
                pi_pulses, rot, pi_backs = self.__routing(ghost_rotation,
                                                          energy_graphs[target_qudit])

                # Map all control levels to physical levels
                new_ctrl_levels = [
                    lp_maps[idx][state]
                    for idx, state in zip(indices, states)
                ]

                # Create new rotation with mapped control levels
                new_parameters = [rot.lev_a, rot.lev_b, rot.theta, rot.phi]
                newr = R(self.circuit,
                         f"Rt{target_qudits}",
                         target_qudit,
                         new_parameters,
                         dimensions[-1],
                         ControlData(indices=indices, ctrl_states=new_ctrl_levels))

                # Return the sequence of operations
                return [op.dag() for op in pi_pulses] + [newr] + [op.dag() for op in pi_backs]

        raise NotImplementedError("The only MULTI gates supported for compilation at "
                                  "the moment are only multi-controlled R gates.")

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions = []

        for gate in reversed(instructions):
            if gate.gate_type == GateTypes.MULTI:
                gate_trans = self.transpile_gate(gate)
                append_to_front(new_instructions, gate_trans)
                # new_instructions.extend(gate_trans)
                gc.collect()
            else:
                append_to_front(new_instructions, gate)
                # new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)

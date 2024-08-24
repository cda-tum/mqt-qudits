from __future__ import annotations

import gc
import typing

import numpy as np

from ....core import NAryTree
from ....exceptions import SequenceFoundException
from ....quantum_circuit import gates
from ....quantum_circuit.components.extensions.gate_types import GateTypes
from ... import CompilerPass
from ...compilation_minitools import new_mod
from ..local_operation_swap import (
    cost_calculator,
    gate_chain_condition,
    graph_rule_ongate,
    graph_rule_update,
)
from ..mapping_aware_transpilation.phy_local_qr_decomp import PhyQrDecomp

if typing.TYPE_CHECKING:
    from ....core import LevelGraph
    from ....core.dfs_tree import Node as TreeNode
    from ....quantum_circuit import QuantumCircuit
    from ....quantum_circuit.gate import Gate
    from ....simulation.backends.backendv2 import Backend

np.seterr(all="ignore")


class PhyLocAdaPass(CompilerPass):
    def __init__(self, backend: Backend) -> None:
        super().__init__(backend)

    def transpile_gate(self, gate: Gate, vrz_prop: bool = False) -> list[Gate]:
        energy_graph_i = self.backend.energy_level_graphs[gate.target_qudits]

        QR = PhyQrDecomp(gate, energy_graph_i)

        _decomp, algorithmic_cost, total_cost = QR.execute()

        Adaptive = PhyAdaptiveDecomposition(
                gate, energy_graph_i, (algorithmic_cost, total_cost), gate._dimensions, Z_prop=vrz_prop
        )
        (matrices_decomposed, _best_cost, new_energy_level_graph) = Adaptive.execute()

        self.backend.energy_level_graphs[gate.target_qudits] = new_energy_level_graph
        return [op.dag() for op in reversed(matrices_decomposed)]

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        self.circuit: QuantumCircuit = circuit
        instructions: list[Gate] = circuit.instructions
        new_instructions = []

        for gate in instructions:
            if gate.gate_type == GateTypes.SINGLE:
                gate_trans = self.transpile_gate(gate)
                new_instructions.extend(gate_trans)
                gc.collect()
            else:
                new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)


class PhyAdaptiveDecomposition:
    def __init__(self, gate: Gate, graph_orig: LevelGraph, cost_limit: tuple[float, float] | None = (0, 0),
                 dimension: int | None = -1, Z_prop: bool | None = False) -> None:
        self.circuit: QuantumCircuit = gate.parent_circuit
        self.U: np.ndarray = gate.to_matrix(identities=0)
        self.qudit_index: int = gate.target_qudits
        self.graph: LevelGraph = graph_orig
        self.graph.phase_storing_setup()
        self.cost_limit: tuple[float, float] = cost_limit
        self.dimension: int = dimension
        self.phase_propagation: bool = Z_prop
        self.TREE: NAryTree = NAryTree()

    def execute(self) -> tuple[list[gates.R | gates.VirtRz], tuple[int, int], LevelGraph]:
        self.TREE.add(
                0,
                gates.CustomOne(
                        self.circuit, "CUo", self.qudit_index, np.identity(self.dimension, dtype="complex"),
                        self.dimension
                ),
                self.U,
                self.graph,
                0,
                0,
                self.cost_limit,
                [],
        )
        try:
            self.DFS(self.TREE.root)
        except SequenceFoundException:
            pass
        finally:
            matrices_decomposed, best_cost, final_graph = self.TREE.retrieve_decomposition(self.TREE.root)

            if matrices_decomposed != []:
                matrices_decomposed, final_graph = self.z_extraction(
                        matrices_decomposed, final_graph, self.phase_propagation
                )
            else:
                pass

            self.TREE.print_tree(self.TREE.root, "TREE: ")

            return matrices_decomposed, best_cost, final_graph

    def z_extraction(self, decomposition: list[TreeNode],
                     placement: LevelGraph, phase_propagation: bool) -> tuple[list[Gate], LevelGraph]:
        matrices = []

        for d in decomposition[1:]:
            # exclude the identity matrix coming from the root of the tree of solutions which is just for correctness
            matrices += d.PI_PULSES
            matrices = [*matrices, d.rotation]

        U_ = decomposition[-1].U_of_level  # take U of last elaboration which should be the diagonal matrix found

        # check if close to diagonal
        Ucopy = U_.copy()

        # check if the diagonal is made only of noise
        valid_diag = any(abs(np.diag(Ucopy)) > 1.0e-4)  # > 1.0e-4

        # are the non diagonal entries zeroed-out
        filtered_Ucopy = abs(Ucopy) > 1.0e-4
        np.fill_diagonal(filtered_Ucopy, 0)

        not_diag = filtered_Ucopy.any()

        if not_diag or not valid_diag:  # if is diagonal enough then somehow signal end of algorithm
            msg = "Matrix isn't close to diagonal!"
            raise Exception(msg)
        diag_U = np.diag(U_)
        dimension = U_.shape[0]

        for i in range(dimension):
            if abs(np.angle(diag_U[i])) > 1.0e-4:
                if phase_propagation:
                    inode = placement.fst_inode
                    if "phase_storage" in placement.nodes[inode]:
                        placement.nodes[i]["phase_storage"] += np.angle(diag_U[i])
                        placement.nodes[i]["phase_storage"] = new_mod(placement.nodes[i]["phase_storage"])
                else:
                    phy_n_i = placement.nodes[i]["lpmap"]

                    phase_gate = gates.VirtRz(
                            self.circuit, "VRz", self.qudit_index, [phy_n_i, np.angle(diag_U[i])], self.dimension
                    )  # old version: VirtRz(np.angle(diag_U[i]), phy_n_i,
                    # dimension)

                    U_ = phase_gate.to_matrix(identities=0) @ U_  # matmul(phase_gate.to_matrix(identities=0), U_)

                    matrices.append(phase_gate)

        if not phase_propagation:
            inode = placement.fst_inode
            if "phase_storage" in placement.nodes[inode]:
                for i in range(len(list(placement.nodes))):
                    thetaZ = new_mod(placement.nodes[i]["phase_storage"])
                    if abs(thetaZ) > 1.0e-4:
                        phase_gate = gates.VirtRz(
                                self.circuit,
                                "VRz",
                                self.qudit_index,
                                [placement.nodes[i]["lpmap"], thetaZ],
                                self.dimension,
                        )  # VirtRz(thetaZ, placement.nodes[i]['lpmap'],
                        # dimension)
                        matrices.append(phase_gate)
                    # reset the node
                    placement.nodes[i]["phase_storage"] = 0

        return matrices, placement

    def DFS(self, current_root: TreeNode, level: int = 0) -> None:
        # check if close to diagonal
        Ucopy = current_root.U_of_level.copy()

        current_placement = current_root.graph

        # is the diagonal noisy?
        valid_diag = any(abs(np.diag(Ucopy)) > 1.0e-4)

        # are the non diagonal entries zeroed-out?
        filtered_Ucopy = abs(Ucopy) > 1.0e-4
        np.fill_diagonal(filtered_Ucopy, 0)

        not_diag = filtered_Ucopy.any()

        # if is diagonal enough then somehow signal end of algorithm
        if (not not_diag) and valid_diag:
            current_root.finished = True

            raise SequenceFoundException(current_root.key)

        ################################################
        ###############
        #########

        # BEGIN SEARCH

        U_ = current_root.U_of_level

        dimension = U_.shape[0]

        for c in range(dimension):
            for r in range(c, dimension):
                for r2 in range(r + 1, dimension):
                    if abs(U_[r2, c]) > 1.0e-8 and (abs(U_[r, c]) > 1.0e-18 or abs(U_[r, c]) == 0):
                        theta = 2 * np.arctan2(abs(U_[r2, c]), abs(U_[r, c]))

                        phi = -(np.pi / 2 + np.angle(U_[r, c]) - np.angle(U_[r2, c]))

                        rotation_involved = gates.R(
                                self.circuit, "R", self.qudit_index, [r, r2, theta, phi], self.dimension
                        )  # R(theta, phi, r, r2, dimension)

                        U_temp = rotation_involved.to_matrix(identities=0) @ U_  # matmul(rotation_involved.matrix, U_)

                        non_zeros = np.count_nonzero(abs(U_temp) > 1.0e-4)

                        (
                            estimated_cost,
                            pi_pulses_routing,
                            new_placement,
                            cost_of_pi_pulses,
                            gate_cost,
                        ) = cost_calculator(rotation_involved, current_placement, non_zeros)

                        next_step_cost = estimated_cost + current_root.current_cost
                        decomp_next_step_cost = cost_of_pi_pulses + gate_cost + current_root.current_decomp_cost

                        branch_condition = (
                                current_root.max_cost[1] - decomp_next_step_cost
                        )  # SECOND POSITION IS PHYSICAL COST
                        # branch_condition_2 = current_root.max_cost[0] - next_step_cost
                        # deprecated: FIRST IS ALGORITHMIC COST

                        if branch_condition > 0 or abs(branch_condition) < 1.0e-12:
                            # if cost is better can be only candidate otherwise try them all

                            self.TREE.global_id_counter += 1
                            new_key = self.TREE.global_id_counter

                            if new_placement.nodes[r]["lpmap"] > new_placement.nodes[r2]["lpmap"]:
                                phi *= -1
                            physical_rotation = gates.R(
                                    self.circuit,
                                    "R",
                                    self.qudit_index,
                                    [new_placement.nodes[r]["lpmap"], new_placement.nodes[r2]["lpmap"], theta, phi],
                                    self.dimension,
                            )
                            # R(theta, phi, new_placement.nodes[r]['lpmap'],
                            # new_placement.nodes[r2]['lpmap'], dimension)
                            #
                            physical_rotation = gate_chain_condition(pi_pulses_routing, physical_rotation)
                            physical_rotation = graph_rule_ongate(physical_rotation, new_placement)

                            # take care of phases accumulated by not pi-pulsing back
                            p_backs = []
                            for ppulse in pi_pulses_routing:
                                p_backs.append(
                                        gates.R(
                                                self.circuit,
                                                "R",
                                                self.qudit_index,
                                                [ppulse.lev_a, ppulse.lev_b, ppulse.theta, -ppulse.phi],
                                                self.dimension,
                                        )
                                )
                                # p_backs.append(R(ppulse.theta, -ppulse.phi, ppulse.lev_a, ppulse.lev_b, dimension))

                            for p_back in p_backs:
                                graph_rule_update(p_back, new_placement)

                            current_root.add(
                                    new_key,
                                    physical_rotation,
                                    U_temp,
                                    new_placement,
                                    next_step_cost,
                                    decomp_next_step_cost,
                                    current_root.max_cost,
                                    pi_pulses_routing,
                            )

        # ===============CONTINUE SEARCH ON CHILDREN========================================
        if current_root.children is not None:
            for child in current_root.children:
                self.DFS(child, level + 1)
        # ===================================================================================

        # END OF RECURSION#

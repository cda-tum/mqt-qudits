from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from ....quantum_circuit import gates
from ... import CompilerPass
from ...compilation_minitools import pi_mod

if TYPE_CHECKING:
    from ....quantum_circuit import QuantumCircuit
    from ....quantum_circuit.gate import Gate
    from ....quantum_circuit.gates import R, VirtRz
    from ....simulation.backends.backendv2 import Backend


class ZPropagationOptPass(CompilerPass):
    def __init__(self, backend: Backend, back: bool | None = True) -> None:
        super().__init__(backend)
        self.back = back

    @staticmethod
    def transpile_gate(gate: Gate) -> Gate:
        return gate

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        return self.remove_z(circuit, self.back)

    @staticmethod
    def propagate_z(circuit: QuantumCircuit, line: list[Gate], back: bool) -> tuple[list[R], list[VirtRz]]:
        z_angles: dict[int, float] = {}
        list_of_x_yrots: list[R] = []
        qudit_index: tuple[int, ...] = line[0].target_qudits
        dimension: int = line[0].dimensions

        for i in range(dimension):
            z_angles[i] = 0.0

        if back:
            line.reverse()

        for gate_index in range(len(line)):
            try:
                line[gate_index].lev_b  # noqa: B018
                # object is R
                if back:
                    new_phi = pi_mod(
                        line[gate_index].phi + z_angles[line[gate_index].lev_a] - z_angles[line[gate_index].lev_b]
                    )
                else:
                    new_phi = pi_mod(
                        line[gate_index].phi - z_angles[line[gate_index].lev_a] + z_angles[line[gate_index].lev_b]
                    )

                list_of_x_yrots.append(
                    gates.R(
                        circuit,
                        "R",
                        qudit_index,
                        [line[gate_index].lev_a, line[gate_index].lev_b, line[gate_index].theta, new_phi],
                        dimension,
                    )
                )
                # list_of_XYrots.append(R(line[gate_index].theta, new_phi,
                # line[gate_index].lev_a, line[gate_index].lev_b, line[gate_index].dimension))
            except AttributeError:  # noqa: PERF203
                try:
                    line[gate_index].lev_a  # noqa: B018
                    # object is VirtRz
                    z_angles[line[gate_index].lev_a] = pi_mod(z_angles[line[gate_index].lev_a] + line[gate_index].phi)
                except AttributeError:
                    pass
        if back:
            list_of_x_yrots.reverse()

        zseq = []
        zseq.extend([
            gates.VirtRz(circuit, "VRz", qudit_index, [e_lev, z_angles[e_lev]], dimension) for e_lev in z_angles
        ])
        # Zseq.append(Rz(Z_angles[e_lev], e_lev, QC.dimension))

        return list_of_x_yrots, zseq

    @staticmethod
    def find_intervals_with_same_target_qudits(instructions: list[Gate]) -> list[tuple[int, ...]]:
        intervals: list[tuple[int, ...]] = []
        current_interval: list[int] = []
        current_target_qudits: tuple[int, ...] | None = None

        for i, instruction in enumerate(instructions):
            target_qudits = instruction.target_qudits

            if current_target_qudits is None or target_qudits == current_target_qudits:
                # If it's the first gate_matrix or the target qudits are the same, add to the current interval
                current_interval.append(i)
            else:
                # If the target qudits are different, save the current interval and start a new one
                intervals.append(tuple(current_interval))
                current_interval = [i]

            current_target_qudits = target_qudits

        # Save the last interval if it exists
        if current_interval:
            intervals.append(tuple(current_interval))

        return intervals

    def remove_z(self, original_circuit: QuantumCircuit, back: bool | None = True) -> QuantumCircuit:
        circuit = original_circuit.copy()
        new_instructions: list[Gate] = copy.deepcopy(circuit.instructions)
        intervals: list[tuple[int, ...]] = self.find_intervals_with_same_target_qudits(circuit.instructions)

        for interval in intervals:
            if len(interval) > 1:
                sequence: list[Gate] = circuit.instructions[interval[0] : interval[-1] + 1]
                fixed_seq: list[R]
                z_tail: list[VirtRz]
                fixed_seq, z_tail = self.propagate_z(circuit, sequence, back)
                """if back:
                    new_instructions[interval[0]: interval[-1] + 1] = z_tail + fixed_seq
                else:
                    new_instructions[interval[0]: interval[-1] + 1] = fixed_seq + z_tail"""
                combined_seq = z_tail + fixed_seq if back else fixed_seq + z_tail
                new_instructions[interval[0] : interval[-1] + 1] = []
                new_instructions.extend(combined_seq)

        return circuit.set_instructions(new_instructions)

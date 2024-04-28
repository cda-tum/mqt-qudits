from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
import tensornetwork as tn

from ...quantum_circuit.gate import Gate, GateTypes
from ..jobs import Job, JobResult
from .backendv2 import Backend
from .stochastic_sim import stochastic_simulation

if TYPE_CHECKING:
    from ...quantum_circuit import QuantumCircuit


class TNSim(Backend):
    def run(self, circuit: QuantumCircuit, **options):
        job = Job(self)

        self._options.update(options)
        self.noise_model = self._options.get("noise_model", None)
        self.shots = self._options.get("shots", 1 if self.noise_model is None else 1000)
        self.memory = self._options.get("memory", False)
        self.full_state_memory = self._options.get("full_state_memory", False)
        self.file_path = self._options.get("file_path", None)
        self.file_name = self._options.get("file_name", None)

        if self.noise_model is not None:
            assert self.shots >= 1000, "Number of shots should be above 1000"
            job.set_result(JobResult(state_vector=self.execute(circuit), counts=stochastic_simulation(self, circuit)))
        else:
            job.set_result(JobResult(state_vector=self.execute(circuit), counts=None))

        return job

    def execute(self, circuit: QuantumCircuit):
        self.system_sizes = circuit.dimensions
        self.circ_operations = circuit.instructions

        result = self.__contract_circuit(self.system_sizes, self.circ_operations)

        result = np.transpose(result.tensor, list(range(len(self.system_sizes))))

        state_size = reduce(operator.mul, self.system_sizes, 1)
        return result.reshape(1, state_size)

    def __init__(self, **fields) -> None:
        self.system_sizes = None
        self.circ_operations = None
        super().__init__(**fields)

    def __apply_gate(self, qudit_edges, gate, operating_qudits) -> None:
        op = tn.Node(gate)
        for i, bit in enumerate(operating_qudits):
            tn.connect(qudit_edges[bit], op[i])
            qudit_edges[bit] = op[i + len(operating_qudits)]

    def __contract_circuit(self, system_sizes, operations: list[Gate]):
        all_nodes = []

        with tn.NodeCollection(all_nodes):
            state_nodes = []
            for s in system_sizes:
                z = [0] * s
                z[0] = 1
                state_nodes.append(tn.Node(np.array(z, dtype="complex")))

            qudits_legs = [node[0] for node in state_nodes]

            for op in operations:
                try:
                    op_matrix = op.to_matrix(identities=1)
                except Exception as e:
                    print(e)
                    op_matrix = op.to_matrix(identities=1)
                lines = op.reference_lines

                if op.gate_type == GateTypes.SINGLE:
                    op_matrix = op_matrix.T
                    # op_matrix = op_matrix.reshape((system_sizes[lines[0]], system_sizes[lines[0]]))

                elif op.gate_type == GateTypes.TWO and not op.is_long_range:
                    op_matrix = op_matrix.T
                    lines = lines.copy()
                    lines.sort()

                    op_matrix = op_matrix.reshape((
                        system_sizes[lines[0]],
                        system_sizes[lines[1]],
                        system_sizes[lines[0]],
                        system_sizes[lines[1]],
                    ))

                elif op.is_long_range or op.gate_type == GateTypes.MULTI:
                    op_matrix = op_matrix.T

                    minimum_line, maximum_line = min(lines), max(lines)
                    interested_lines = list(range(minimum_line, maximum_line + 1))
                    inputs_outputs_legs = []
                    for i in interested_lines:
                        inputs_outputs_legs.append(system_sizes[i])
                    inputs_outputs_legs += inputs_outputs_legs

                    op_matrix = op_matrix.reshape(tuple(inputs_outputs_legs))
                    lines = interested_lines

                self.__apply_gate(qudits_legs, op_matrix, lines)

        return tn.contractors.auto(all_nodes, output_edge_order=qudits_legs)

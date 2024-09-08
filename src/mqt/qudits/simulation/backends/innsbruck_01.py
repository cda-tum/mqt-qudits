from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import Unpack

from ...core import LevelGraph
from ..jobs import Job, JobResult
from .backendv2 import Backend

if TYPE_CHECKING:
    from ...quantum_circuit import QuantumCircuit
    from .. import MQTQuditProvider
    from ..noise_tools import NoiseModel


class Innsbruck01(Backend):
    @property
    def version(self) -> int:
        return 0

    def __init__(
        self,
        provider: MQTQuditProvider,
        **fields: Unpack[Backend.DefaultOptions],
    ) -> None:
        super().__init__(
            provider=provider,
            name="Innsbruck01",
            description="Interface to the Innsbruck machine 01.",
            **fields,
        )
        self.outcome = -1
        self.options["noise_model"] = self.__noise_model()
        self.author = "<Kevin Mato>"
        self._energy_level_graphs: list[LevelGraph] = []

    @property
    def energy_level_graphs(self) -> list[LevelGraph]:
        if len(self._energy_level_graphs) == 0:
            e_graphs: list[LevelGraph] = []
            # declare the edges on the energy level graph between logic states .
            edges = [
                (2, 0, {"delta_m": 0, "sensitivity": 3}),
                (3, 0, {"delta_m": 0, "sensitivity": 3}),
                (4, 0, {"delta_m": 0, "sensitivity": 4}),
                (5, 0, {"delta_m": 0, "sensitivity": 4}),
                (1, 2, {"delta_m": 0, "sensitivity": 4}),
                (1, 3, {"delta_m": 0, "sensitivity": 3}),
                (1, 4, {"delta_m": 0, "sensitivity": 3}),
                (1, 5, {"delta_m": 0, "sensitivity": 3}),
            ]
            # name explicitly the logic states .
            nodes = [0, 1, 2, 3, 4, 5]
            # declare physical levels in order of mapping of the logic states just declared .
            # i.e. here we will have Logic 0 -> Phys. 0, have Logic 1 -> Phys. 1, have Logic 2 -> Phys. 2 .
            nmap = [0, 1, 2, 3, 4, 5]
            # Construct the qudit energy level graph, the last field is the list of logic state that are used for the
            # calibrations of the operations. note: only the first is one counts in our current cost function.
            graph_0 = LevelGraph(edges, nodes, nmap, [1])
            # declare the edges on the energy level graph between logic states .
            edges_1 = [
                (2, 0, {"delta_m": 0, "sensitivity": 3}),
                (3, 0, {"delta_m": 0, "sensitivity": 3}),
                (4, 0, {"delta_m": 0, "sensitivity": 4}),
                (5, 0, {"delta_m": 0, "sensitivity": 4}),
                (1, 2, {"delta_m": 0, "sensitivity": 4}),
                (1, 3, {"delta_m": 0, "sensitivity": 3}),
                (1, 4, {"delta_m": 0, "sensitivity": 3}),
                (1, 5, {"delta_m": 0, "sensitivity": 3}),
            ]
            # name explicitly the logic states .
            nodes_1 = [0, 1, 2, 3, 4, 5]
            # declare physical levels in order of mapping of the logic states just declared .
            # i.e. here we will have Logic 0 -> Phys. 0, have Logic 1 -> Phys. 1, have Logic 2 -> Phys. 2 .
            nmap_1 = [0, 1, 2, 3, 4, 5]
            # Construct the qudit energy level graph, the last field is the list of logic state that are used for the
            # calibrations of the operations. note: only the first is one counts in our current cost function.
            graph_1 = LevelGraph(edges_1, nodes_1, nmap_1, [1])

            # declare the edges on the energy level graph between logic states .
            edges_2 = [
                (2, 0, {"delta_m": 0, "sensitivity": 3}),
                (3, 0, {"delta_m": 0, "sensitivity": 3}),
                (4, 0, {"delta_m": 0, "sensitivity": 4}),
                (5, 0, {"delta_m": 0, "sensitivity": 4}),
                (1, 2, {"delta_m": 0, "sensitivity": 4}),
                (1, 3, {"delta_m": 0, "sensitivity": 3}),
                (1, 4, {"delta_m": 0, "sensitivity": 3}),
                (1, 5, {"delta_m": 0, "sensitivity": 3}),
            ]
            # name explicitly the logic states .
            nodes_2 = [0, 1, 2, 3, 4, 5]
            # declare physical levels in order of mapping of the logic states just declared .
            # i.e. here we will have Logic 0 -> Phys. 0, have Logic 1 -> Phys. 1, have Logic 2 -> Phys. 2 .
            nmap_2 = [0, 1, 2, 3, 4, 5]
            # Construct the qudit energy level graph, the last field is the list of logic state that are used for the
            # calibrations of the operations. note: only the first is one counts in our current cost function.
            graph_2 = LevelGraph(edges_2, nodes_2, nmap_2, [1])

            e_graphs.extend((graph_0, graph_1, graph_2))

            self._energy_level_graphs = e_graphs
        return self._energy_level_graphs

    def __noise_model(self) -> NoiseModel | None:
        return self.noise_model

    def run(self, circuit: QuantumCircuit, **options: Unpack[Backend.DefaultOptions]) -> Job:
        job = Job(self)

        self._options.update(options)
        self.noise_model = self._options.get("noise_model", None)
        self.shots = self._options.get("shots", 50)
        self.memory = self._options.get("memory", False)
        self.full_state_memory = self._options.get("full_state_memory", False)
        self.file_path = self._options.get("file_path", None)
        self.file_name = self._options.get("file_name", None)

        assert self.shots >= 50, "Number of shots should be above 50"
        collected_counts: list[int] = []
        for _ in range(self.shots):
            self.execute(circuit)
            collected_counts.append(self.outcome)
        job.set_result(JobResult(state_vector=np.array([]), counts=collected_counts))

        return job

    def execute(self, circuit: QuantumCircuit, noise_model: NoiseModel | None = None) -> None:
        _ = noise_model  # Silences the unused argument warning
        self.system_sizes = circuit.dimensions
        self.circ_operations = circuit.instructions
        self.outcome = -1

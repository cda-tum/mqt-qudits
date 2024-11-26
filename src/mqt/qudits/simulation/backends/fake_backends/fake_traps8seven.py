from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Unpack

from ....core import LevelGraph
from ...noise_tools import Noise, NoiseModel
from ..tnsim import TNSim

if TYPE_CHECKING:
    from ... import MQTQuditProvider
    from ..backendv2 import Backend


class FakeIonTraps8Seven(TNSim):
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
                name="FakeTrap8Seven",
                description="A Fake backend of an ion trap qudit machine with 8 ions and 7 levels each",
                **fields,
        )
        self.options["noise_model"] = self.__noise_model()
        self.author = "<Kevin Mato>"
        self._energy_level_graphs: list[LevelGraph] = []

    @property
    def energy_level_graphs(self) -> list[LevelGraph]:
        if len(self._energy_level_graphs) == 0:
            e_graphs: list[LevelGraph] = []

            # Create graphs for all 8 ions
            for i in range(8):
                # declare the edges on the energy level graph between logic states
                # we have to fake a direct connection between 0 and 1 to make transpilations
                # physically compatible also for small systems
                edges = [
                    (0, 1, {"delta_m": 0, "sensitivity": 4}),
                    (2, 0, {"delta_m": 0, "sensitivity": 3}),
                    (3, 0, {"delta_m": 0, "sensitivity": 3}),
                    (4, 0, {"delta_m": 0, "sensitivity": 3}),
                    (5, 0, {"delta_m": 0, "sensitivity": 4}),
                    (6, 0, {"delta_m": 0, "sensitivity": 4}),
                    (1, 2, {"delta_m": 0, "sensitivity": 4}),
                    (1, 3, {"delta_m": 0, "sensitivity": 3}),
                    (1, 4, {"delta_m": 0, "sensitivity": 3}),
                    (1, 5, {"delta_m": 0, "sensitivity": 4}),
                    (1, 6, {"delta_m": 0, "sensitivity": 4}),
                ]

                # name explicitly the logic states (0 through 6 for seven levels)
                nodes = list(range(7))

                # Different mappings for different ions
                if i % 3 == 0:
                    nmap = [1, 0, 2, 3, 4, 5, 6]  # Similar to graph_0 in original
                elif i % 3 == 1:
                    nmap = [2, 1, 0, 3, 4, 5, 6]  # Similar to graph_1 in original
                else:
                    nmap = [0, 2, 1, 3, 4, 5, 6]  # Similar to graph_2 in original

                # Construct the qudit energy level graph
                graph = LevelGraph(edges, nodes, nmap, [1])
                e_graphs.append(graph)

            self._energy_level_graphs = e_graphs
        return self._energy_level_graphs

    def __noise_model(self) -> NoiseModel:
        # Using similar noise parameters as the original
        local_error = Noise(probability_depolarizing=0.001, probability_dephasing=0.001)
        local_error_rz = Noise(probability_depolarizing=0.03, probability_dephasing=0.03)
        entangling_error = Noise(probability_depolarizing=0.1, probability_dephasing=0.001)
        entangling_error_extra = Noise(probability_depolarizing=0.1, probability_dephasing=0.1)
        entangling_error_on_target = Noise(probability_depolarizing=0.1, probability_dephasing=0.0)
        entangling_error_on_control = Noise(probability_depolarizing=0.01, probability_dephasing=0.0)

        noise_model = NoiseModel()

        # Add errors to noise model (same as original)
        noise_model.add_all_qudit_quantum_error(local_error, ["csum"])
        noise_model.add_nonlocal_quantum_error(entangling_error, ["cx", "ls", "ms"])
        noise_model.add_nonlocal_quantum_error_on_target(entangling_error_on_target, ["cx", "ls", "ms"])
        noise_model.add_nonlocal_quantum_error_on_control(entangling_error_on_control, ["csum", "cx", "ls", "ms"])
        noise_model.add_nonlocal_quantum_error(entangling_error_extra, ["csum"])
        noise_model.add_quantum_error_locally(local_error, ["h", "rxy", "s", "x", "z"])
        noise_model.add_quantum_error_locally(local_error_rz, ["rz", "virtrz"])

        self.noise_model = noise_model
        return noise_model

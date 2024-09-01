from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...core import LevelGraph
    from ...quantum_circuit import QuantumCircuit
    from .. import MQTQuditProvider
    from ..jobs import Job
    from ..noise_tools import NoiseModel


class Backend(ABC):
    def __init__(
        self,
        provider: MQTQuditProvider | None = None,
        name: str | None = None,
        description: str | None = None,
        **fields: dict[str, Any],
    ) -> None:
        self._provider = provider
        self.name = name
        self.description: str = description
        self._energy_level_graphs = []
        self.noise_model: NoiseModel | None = None
        self.shots: int = 50
        self.memory: bool = False
        self.full_state_memory: bool = False
        self.file_path: str | None = None
        self.file_name: str | None = None

        self._options: Any = self._default_options()
        if fields:
            self._options.update(fields)

    def __noise_model(self) -> NoiseModel:
        return self.noise_model

    @property
    def energy_level_graphs(self) -> list[LevelGraph, LevelGraph]:
        raise NotImplementedError

    def _default_options(self) -> dict[str, int | bool | NoiseModel | None]:
        return {"shots": 50, "memory": False, "noise_model": self.__noise_model()}

    def set_options(self, **fields: Any) -> None:  # noqa: ANN401
        for field in fields:
            if not hasattr(self._options, field):
                msg = f"Options field '{field}' is not valid for this backend"
                raise AttributeError(msg)
        self._options.update_options(**fields)

    @property
    def options(self) -> dict[str, Any]:
        return self._options

    @property
    def provider(self) -> MQTQuditProvider:
        return self._provider

    @abstractmethod
    def run(self, circuit: QuantumCircuit, **options: Any) -> Job:  # noqa: ANN401
        pass

    def execute(self, circuit: QuantumCircuit, noise_model: NoiseModel | None = None) -> None:
        raise NotImplementedError

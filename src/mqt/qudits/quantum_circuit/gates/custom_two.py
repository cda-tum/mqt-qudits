from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class CustomTwo(Gate):
    """Two body custom gate"""

    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int] | int,
        parameters: np.ndarray,
        dimensions: list[int] | int,
        controls: ControlData | None = None,
    ) -> None:
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.TWO,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
            params=parameters,
        )
        if self.validate_parameter(parameters):
            self.__array_storage = parameters
        self.qasm_tag = "cutwo"

    def __array__(self) -> np.ndarray:
        return self.__array_storage

    def validate_parameter(self, parameter=None) -> bool:
        return isinstance(parameter, np.ndarray)

    def __str__(self) -> str:
        # TODO
        pass

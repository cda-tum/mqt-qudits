from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING

import numpy as np
from ..gate import Gate
from ..components.extensions.gate_types import GateTypes

if TYPE_CHECKING:
    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class Perm(Gate):
    def __init__(
            self,
            circuit: QuantumCircuit,
            name: str,
            target_qudits: list[int] | int,
            parameters: list,
            dimensions: list[int] | int,
            controls: ControlData | None = None,
    ) -> None:
        super().__init__(
                circuit=circuit,
                name=name,
                gate_type=GateTypes.MULTI,
                target_qudits=target_qudits,
                dimensions=dimensions,
                control_set=controls,
        )
        if self.validate_parameter(parameters):
            self.perm_data = parameters
            self._params = parameters
        self.qasm_tag = "pm"

    def __array__(self) -> np.ndarray:
        return np.eye(reduce(operator.mul, self._dimensions))[:, self.perm_data]

    def validate_parameter(self, parameter) -> bool:
        assert isinstance(parameter, list), "Input is not a list"
        num_nums = reduce(operator.mul, self._dimensions)
        assert all(
                (0 <= num < len(parameter) and num < num_nums) for num in parameter
        ), "Numbers are not within the range of the list length"
        return True

    def __str__(self) -> str:
        # TODO
        pass

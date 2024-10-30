from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate
from .gellmann import GellMann

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData
    from ..gate import Parameter


class EachX(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: int,
        parameters: list[int],
        dimensions: int,
        controls: ControlData | None = None,
    ) -> None:
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.SINGLE,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
            qasm_tag="eachx",
        )

        if self.validate_parameter(parameters):
            self.original_lev_a: int = parameters[0]
            self.original_lev_b: int = parameters[1]
            self.lev_a, self.lev_b = self.levels_setter(self.original_lev_a, self.original_lev_b)
            self._params = parameters

    def __array__(self) -> NDArray:  # noqa: PLW3201
        dimension = self.dimensions
        matrix = np.identity(dimension, dtype="complex")

        matrix[self.lev_a, self.lev_a] = 0.0
        matrix[self.lev_b, self.lev_b] = 0.0
        matrix[self.lev_a, self.lev_b] = 1.0
        matrix[self.lev_b, self.lev_a] = 1.0

        return matrix

    @staticmethod
    def levels_setter(la: int, lb: int) -> tuple[int, int]:
        if la < lb:
            return la, lb
        return lb, la

    def validate_parameter(self, parameter: Parameter) -> bool:
        if parameter is None:
            return False

        if isinstance(parameter, list):
            assert isinstance(parameter[0], int)
            assert isinstance(parameter[1], int)
            assert parameter[0] >= 0
            assert parameter[0] < self.dimensions
            assert parameter[1] >= 0
            assert parameter[1] < self.dimensions
            assert parameter[0] != parameter[1]
            # Useful to remember direction of the rotation
            self.original_lev_a = parameter[0]
            self.original_lev_b = parameter[1]

            return True

        if isinstance(parameter, np.ndarray):
            # Add validation for numpy array if needed
            return False

        return False

    @property
    def dimensions(self) -> int:
        assert isinstance(self._dimensions, int), "Dimensions must be an integer"
        return self._dimensions

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class CustomOne(Gate):
    """One body custom gate."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: int,
        parameters: NDArray,
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
            params=parameters,
            qasm_tag="cuone",
        )
        self.__array_storage: NDArray = None
        if self.validate_parameter(parameters):
            self.__array_storage = parameters

    def __array__(self) -> NDArray:  # noqa: PLW3201
        return  self.__array_storage

    @staticmethod
    def validate_parameter(parameter: NDArray | None = None) -> bool:
        if parameter is None:
            return True  # or False, depending on whether None is considered valid
        return isinstance(parameter, np.ndarray) and (
            parameter.dtype == np.complex128 or np.issubdtype(parameter.dtype, np.number)
        )

    def _dagger_properties(self) -> None:
        self.__array_storage = self.__array_storage.conj().T
        self.update_params(self.__array_storage)

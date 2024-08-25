from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData

import operator

from scipy.stats import unitary_group


class RandU(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int],
        dimensions: list[int],
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
        self.qasm_tag = "rdu"

    def __array__(self) -> NDArray:  # noqa: PLW3201
        dim = reduce(operator.mul, self._dimensions)
        return unitary_group.rvs(dim)

    @staticmethod
    def validate_parameter() -> bool:
        return True

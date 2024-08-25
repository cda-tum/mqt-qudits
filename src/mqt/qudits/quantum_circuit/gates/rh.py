from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate
from .r import R

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class Rh(Gate):
    """SU2 Hadamard"""

    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: int,
        parameters: list[int],
        dimensions: list[int],
        controls: ControlData | None = None,
    ) -> None:
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.SINGLE,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
        )
        self.original_lev_b = None
        self.original_lev_a = None
        if self.validate_parameter(parameters):
            self.lev_a, self.lev_b = parameters
            self.lev_a, self.lev_b = self.levels_setter(self.original_lev_a, self.original_lev_b)
            self._params = parameters
        self.qasm_tag = "rh"

    def __array__(self) -> NDArray:  # noqa: PLW3201
        # (R(-np.pi, 0, l1, l2, dim) * R(np.pi / 2, np.pi / 2, l1, l2, dim))
        dimension = self._dimensions

        pi_x = R(
            self.parent_circuit, "R", self.target_qudits, [self.lev_a, self.lev_b, -np.pi, 0.0], dimension
        ).to_matrix()
        rotate = R(
            self.parent_circuit,
            "R",
            self.target_qudits,
            [self.lev_a, self.lev_b, np.pi / 2, np.pi / 2],
            dimension,
        ).to_matrix()

        return pi_x @ rotate

    @staticmethod
    def levels_setter(la: int, lb: int) -> tuple[int, int]:
        if la < lb:
            return la, lb
        return lb, la

    def validate_parameter(self, parameter: list[int]) -> bool:
        assert isinstance(parameter[0], int)
        assert isinstance(parameter[1], int)

        assert parameter[0] >= 0
        assert parameter[0] < self._dimensions
        assert parameter[1] >= 0
        assert parameter[1] < self._dimensions
        assert parameter[0] != parameter[1]
        # Useful to remember direction of the rotation
        self.original_lev_a = parameter[0]
        self.original_lev_b = parameter[1]

        return True

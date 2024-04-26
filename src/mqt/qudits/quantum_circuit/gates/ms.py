from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

from ..components.extensions.gate_types import GateTypes
from ..gate import Gate
from .gellmann import GellMann

if TYPE_CHECKING:
    from ..circuit import QuantumCircuit
    from ..components.extensions.controls import ControlData


class MS(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int] | int,
        parameters: list | None,
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
        )
        if self.validate_parameter(parameters):
            self.theta = parameters[0]
            self._params = parameters
        self.qasm_tag = "ms"

    def __array__(self) -> np.ndarray:
        theta = self.theta
        dimension_0 = self._dimensions[0]
        dimension_1 = self._dimensions[1]

        return expm(
            -1j
            * theta
            * (
                (
                    np.outer(
                        np.identity(dimension_0, dtype="complex"),
                        GellMann(
                            self.parent_circuit,
                            "Gellman_s",
                            self._target_qudits,
                            [0, 1, "s"],
                            dimension_1,
                            None,
                        ).to_matrix(),
                    )
                    + np.outer(
                        GellMann(
                            self.parent_circuit,
                            "Gellman_s",
                            self._target_qudits,
                            [0, 1, "s"],
                            dimension_0,
                            None,
                        ).to_matrix(),
                        np.identity(dimension_1, dtype="complex"),
                    )
                )
                @ (
                    np.outer(
                        np.identity(dimension_0, dtype="complex"),
                        GellMann(
                            self.parent_circuit,
                            "Gellman_s",
                            self._target_qudits,
                            [0, 1, "s"],
                            dimension_1,
                            None,
                        ).to_matrix(),
                    )
                    + np.outer(
                        GellMann(
                            self.parent_circuit,
                            "Gellman_s",
                            self._target_qudits,
                            [0, 1, "s"],
                            dimension_0,
                            None,
                        ).to_matrix(),
                        np.identity(dimension_1, dtype="complex"),
                    )
                )
            )
            / 4
        )

    def validate_parameter(self, parameter) -> bool:
        assert 0 <= parameter[0] <= 2 * np.pi, f"Angle should be in the range [0, 2*pi]: {parameter[0]}"
        return True

    def __str__(self) -> str:
        # TODO
        pass

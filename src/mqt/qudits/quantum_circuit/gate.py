from __future__ import annotations

import random
import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mqt.qudits.quantum_circuit.components.extensions.matrix_factory import MatrixFactory

from ..exceptions import CircuitError
from .components.extensions.controls import ControlData
from .components.extensions.gate_types import GateTypes

if TYPE_CHECKING:
    import enum

    from .circuit import QuantumCircuit


class Instruction(ABC):
    @abstractmethod
    def __init__(self, name: str) -> None:
        pass


class Gate(Instruction):
    """Unitary gate_matrix."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        gate_type: enum,
        target_qudits: list[int] | int,
        dimensions: list[int] | int,
        params: list | np.ndarray | None = None,
        control_set=None,
        label: str | None = None,
        duration=None,
        unit="dt",
    ) -> None:
        self.dagger = False
        self.parent_circuit = circuit
        self._name = name
        self.gate_type = gate_type
        self._target_qudits = target_qudits
        self._dimensions = dimensions
        self._params = params
        self._label = label
        self._duration = duration
        self._unit = unit
        self._controls_data = None
        self.is_long_range = self.check_long_range()
        if control_set:
            self.control(**vars(control_set))
        self.qasm_tag = ""

    @property
    def reference_lines(self):
        lines = []
        if isinstance(self.target_qudits, int):
            lines = self.get_control_lines.copy()
            lines.append(self.target_qudits)
        elif isinstance(self.target_qudits, list):
            lines = self.get_control_lines.copy() + self.target_qudits.copy()
        if len(lines) == 0:
            msg = "Gate has no target or control lines"
            raise CircuitError(msg)
        return lines

    @abstractmethod
    def __array__(self) -> np.ndarray:
        pass

    def dag(self):
        self._name += "_dag"
        self.dagger = True
        return self

    def to_matrix(self, identities=0) -> np.ndarray:
        """Return a np.ndarray for the gate_matrix unitary parameters.

        Returns:
            np.ndarray: if the Gate subclass has a parameters definition.

        Raises:
            CircuitError: If a Gate subclass does not implement this method an
                exception will be raised when this base class method is called.
        """
        if hasattr(self, "__array__"):
            matrix_factory = MatrixFactory(self, identities)
            return matrix_factory.generate_matrix()
        msg = "to_matrix not defined for this "
        raise CircuitError(msg, {type(self)})

    def control(self, indices: list[int] | int, ctrl_states: list[int] | int):
        if len(indices) == 0 or len(ctrl_states) == 0:
            return self
        # AT THE MOMENT WE SUPPORT CONTROL OF SINGLE QUDIT GATES
        assert self.gate_type == GateTypes.SINGLE
        if len(indices) > self.parent_circuit.num_qudits or any(
            idx >= self.parent_circuit.num_qudits for idx in indices
        ):
            msg = "Indices or Number of Controls is beyond the Quantum Circuit Size"
            raise IndexError(msg)
        if isinstance(self.target_qudits, int):
            if self.target_qudits in indices:
                msg = "Controls overlap with targets"
                raise IndexError(msg)
        elif any(idx in list(self.target_qudits) for idx in indices):
            msg = "Controls overlap with targets"
            raise IndexError(msg)
        # if isinstance(self._dimensions, int):
        #    dimensions = [self._dimensions]
        if any(ctrl >= self.parent_circuit.dimensions[i] for i, ctrl in enumerate(ctrl_states)):
            msg = "Controls States beyond qudit size "
            raise IndexError(msg)
        self._controls_data = ControlData(indices, ctrl_states)
        # Set new type
        if len(self.reference_lines) == 2:
            self.set_gate_type_two()
        elif len(self.reference_lines) > 2:
            self.set_gate_type_multi()
        self.check_long_range()
        return self

    @abstractmethod
    def validate_parameter(self, parameter):
        pass

    @property
    def target_qudits(self) -> list[int] | np.ndarray:
        """
        Get the target qudits.

        Returns:
            Union[List[int], np.ndarray]: The current target qudits.
        """
        return self._target_qudits

    @target_qudits.setter
    def target_qudits(self, value: list[int] | np.ndarray) -> None:
        """
        Set the target qudits.

        Args:
            value (Union[List[int], np.ndarray]): The new target qudits.

        Raises:
            ValueError: If the input is not a list of integers or numpy array of integers.
        """
        if (isinstance(value, list) and all(isinstance(x, int) for x in value)) or (isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.integer)):
            self._target_qudits = value
        else:
            msg = "target_qudits must be a list of integers or a numpy array of integers"
            raise ValueError(msg)

    def __qasm__(self) -> str:
        """Generate QASM for Gate export"""
        string = f"{self.qasm_tag} "
        if isinstance(self._params, np.ndarray):
            string += self.return_custom_data()
        elif self._params:
            string += "("
            for parameter in self._params:
                string += f"{parameter}, "
            string = string[:-2]
            string += ") "
        if isinstance(self.target_qudits, int):
            targets = [self.target_qudits]
        elif isinstance(self.target_qudits, list):
            targets = self.target_qudits
        for qudit in targets:
            string += (
                f"{self.parent_circuit.inverse_sitemap[qudit][0]}[{self.parent_circuit.inverse_sitemap[qudit][1]}], "
            )
        string = string[:-2]
        if self._controls_data:
            string += " ctl "
            for _ctrl in self._controls_data.indices:
                string += (
                    f"{self.parent_circuit.inverse_sitemap[qudit][0]}[{self.parent_circuit.inverse_sitemap[qudit][1]}] "
                )
            string += str(self._controls_data.ctrl_states)

        return string + ";\n"

    @abstractmethod
    def __str__(self) -> str:
        # String representation for drawing?
        pass

    def check_long_range(self):
        target_qudits = self.reference_lines
        if isinstance(target_qudits, list) and len(target_qudits) > 0:
            self.is_long_range = any((b - a) > 1 for a, b in zip(sorted(target_qudits)[:-1], sorted(target_qudits)[1:]))
        return self.is_long_range

    def set_gate_type_single(self) -> None:
        self.gate_type = GateTypes.SINGLE

    def set_gate_type_two(self) -> None:
        self.gate_type = GateTypes.TWO

    def set_gate_type_multi(self) -> None:
        self.gate_type = GateTypes.MULTI

    @property
    def get_control_lines(self):
        if self._controls_data:
            return self._controls_data.indices
        return []

    @property
    def control_info(self):
        return {
            "target": self.target_qudits,
            "dimensions_slice": self._dimensions,
            "params": self._params,
            "controls": self._controls_data,
        }

    def return_custom_data(self) -> str:
        if not self.parent_circuit.path_save:
            return "(custom_data) "

        key = "".join(random.choice(string.ascii_letters) for _ in range(4))
        file_path = Path(self.parent_circuit.path_save) / f"{self._name}_{key}.npy"
        np.save(file_path, self._params)
        return f"({file_path}) "

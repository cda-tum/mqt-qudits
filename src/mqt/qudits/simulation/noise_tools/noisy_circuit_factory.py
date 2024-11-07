from __future__ import annotations

import copy
import os
import time
from functools import partial
from itertools import combinations, product
from typing import TYPE_CHECKING, cast

import numpy as np

from ...quantum_circuit import QuantumCircuit
from ...quantum_circuit.components.extensions.gate_types import GateTypes
from .noise import Noise, SubspaceNoise

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.random import Generator

    from ...quantum_circuit.gate import Gate
    from .noise import NoiseModel


class NoisyCircuitFactory:
    def __init__(self, noise_model: NoiseModel, circuit: QuantumCircuit) -> None:
        self.noise_model: NoiseModel = noise_model
        self.circuit: QuantumCircuit = circuit
        self.rng: Generator = self._initialize_rng()

    @staticmethod
    def _initialize_rng() -> Generator:
        current_time = int(time.time() * 1000)
        seed = hash((os.getpid(), current_time)) % 2**32
        return np.random.default_rng(seed)

    def generate_circuit(self) -> QuantumCircuit:
        noisy_circuit = QuantumCircuit(self.circuit.num_qudits, self.circuit.dimensions, self.circuit.num_cl)
        noisy_circuit.number_gates = 0

        for instruction in self.circuit.instructions:
            copied_instruction = copy.deepcopy(instruction)
            noisy_circuit.instructions.append(copied_instruction)
            noisy_circuit.number_gates += 1

            self._apply_noise(noisy_circuit, instruction)

        return noisy_circuit

    def _apply_noise(self, noisy_circuit: QuantumCircuit, instruction: Gate) -> None:
        if instruction.qasm_tag not in self.noise_model.quantum_errors:
            return

        for mode, noise_info in self.noise_model.quantum_errors[instruction.qasm_tag].items():
            qudits = self._get_affected_qudits(instruction, mode)
            if qudits is None:
                continue  # type: ignore[unreachable]

            self._apply_depolarizing_noise(noisy_circuit, qudits, noise_info)
            self._apply_dephasing_noise(noisy_circuit, qudits, noise_info)

    def _get_affected_qudits(self, instruction: Gate, mode: str) -> list[int]:
        if isinstance(mode, str):
            return self._get_qudits_for_mode(instruction, mode)
        msg = "Something broken is constructrion of Noise Model."  # type: ignore[unreachable]
        raise ValueError(msg)

    def _get_qudits_for_mode(self, instruction: Gate, mode: str) -> list[int]:
        mode_handlers: dict[str, Callable[[], list[int]]] = {
            "local": lambda: instruction.reference_lines,
            "all": lambda: list(range(instruction.parent_circuit.num_qudits)),
            "nonlocal": partial(self._get_nonlocal_qudits, instruction),
            "control": partial(self._get_control_qudits, instruction),
            "target": partial(self._get_target_qudits, instruction),
        }

        handler = mode_handlers.get(mode)
        if not handler:
            msg = f"Unknown mode: {mode}"
            raise ValueError(msg)

        return handler()

    @staticmethod
    def _get_nonlocal_qudits(instruction: Gate) -> list[int]:
        if instruction.gate_type not in {GateTypes.TWO, GateTypes.MULTI}:
            msg = f"Nonlocal mode not applicable for gate type: {instruction.gate_type}"
            raise ValueError(msg)
        return instruction.reference_lines

    def _get_control_qudits(self, instruction: Gate) -> list[int]:
        self._validate_two_qudit_gate(instruction, "Control")
        qudits_targeted = cast(list[int], instruction.target_qudits)
        return qudits_targeted[:1]

    def _get_target_qudits(self, instruction: Gate) -> list[int]:
        self._validate_two_qudit_gate(instruction, "Target")
        qudits_targeted = cast(list[int], instruction.target_qudits)
        return qudits_targeted[1:]

    @staticmethod
    def _validate_two_qudit_gate(instruction: Gate, mode: str) -> None:
        if instruction.gate_type != GateTypes.TWO:
            msg = f"{mode} mode only applicable for two-qudit gates, not {instruction.gate_type}"
            raise ValueError(msg)

    def _apply_depolarizing_noise(
        self, noisy_circuit: QuantumCircuit, qudits: list[int], noise_info: Noise | SubspaceNoise
    ) -> None:
        if isinstance(noise_info, Noise):  # Mathematical Noise
            for dit in qudits:
                dim = noisy_circuit.dimensions[dit]
                prob_each = noise_info.probability_depolarizing / dim / dim
                n_xz = list(product(range(dim), repeat=2))
                p_xz = [1 - prob_each * (dim * dim - 1)] + [prob_each] * (dim * dim - 1)
                n_x, n_z = self.rng.choice(n_xz, p=p_xz)
                for _ in range(n_x):
                    noisy_circuit.x(dit)
                for _ in range(n_z):
                    noisy_circuit.z(dit)
        elif isinstance(noise_info, SubspaceNoise):  # Physical Noise
            for dit in qudits:
                dim = noisy_circuit.dimensions[dit]
                levels = combinations(range(dim), 2)
                for lev_tuple in levels:
                    if lev_tuple in noise_info.probs:
                        prob_each = noise_info.probs[lev_tuple].probability_depolarizing / 4
                        n_xz = list(product(range(2), repeat=2))
                        p_xz = [1 - prob_each * 3] + [prob_each] * 3
                        n_x, n_z = self.rng.choice(n_xz, p=p_xz)
                        if n_x:
                            noisy_circuit.noisex(dit, list(lev_tuple))
                        if n_z:
                            noisy_circuit.noisez(dit, lev_tuple[1])

    def _apply_dephasing_noise(
        self, noisy_circuit: QuantumCircuit, qudits: list[int], noise_info: Noise | SubspaceNoise
    ) -> None:
        if isinstance(noise_info, Noise):  # Mathematical Noise
            for dit in qudits:
                dim = noisy_circuit.dimensions[dit]
                prob_each = noise_info.probability_dephasing / dim
                n_z = list(range(dim))
                p_z = [1 - prob_each * (dim - 1)] + [prob_each] * (dim - 1)
                level = self.rng.choice(n_z, p=p_z)
                if level != 0:
                    noisy_circuit.noisez(dit, level)
        elif isinstance(noise_info, SubspaceNoise):  # Physical Noise
            for dit in qudits:
                dim = noisy_circuit.dimensions[dit]
                levels = combinations(range(dim), 2)
                for lev_tuple in levels:
                    if lev_tuple in noise_info.probs:
                        list(set(range(dim)) - set(lev_tuple))
                        prob_each = noise_info.probs[lev_tuple].probability_dephasing / dim
                        n_z = list(range(dim))
                        p_z = [1 - prob_each * (dim - 1)] + [prob_each] * (dim - 1)
                        level = self.rng.choice(n_z, p_z)
                        if level not in {0, lev_tuple[0], lev_tuple[1]}:
                            noisy_circuit.noisez(dit, level)

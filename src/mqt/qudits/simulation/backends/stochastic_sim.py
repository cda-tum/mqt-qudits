from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import TYPE_CHECKING

import numpy as np

from ..noise_tools import NoiseModel, NoisyCircuitFactory
from ..save_info import save_full_states, save_shots

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...quantum_circuit import QuantumCircuit
    from .backendv2 import Backend


def generate_seed() -> int:
    """Generate a random seed for numpy random generator."""
    current_time = int(time.time() * 1000)
    return hash((os.getpid(), current_time)) % 2 ** 32


def measure_state(vector_data: NDArray[np.complex128]) -> int:
    """Measure the quantum state and return the result."""
    probabilities = np.abs(vector_data.ravel()) ** 2
    rng = np.random.default_rng(generate_seed())
    return rng.choice(len(probabilities), p=probabilities)


def save_results(backend: Backend, results: list[int | NDArray[np.complex128]]) -> None:
    """Save the simulation results based on backend configuration."""
    if backend.full_state_memory or backend.memory:
        save_function = save_full_states if backend.full_state_memory else save_shots
        save_function(results, backend.file_path, backend.file_name)


def stochastic_simulation(backend: Backend, circuit: QuantumCircuit) -> NDArray | list[int]:
    shots = backend.shots
    num_processes = mp.cpu_count()
    from . import MISim, TNSim
    if isinstance(backend, TNSim):
        factory = NoisyCircuitFactory(backend.noise_model, circuit)
        args = [(backend, factory) for _ in range(shots)]
    elif isinstance(backend, MISim):
        args = [(backend, (circuit, backend.noise_model)) for _ in range(shots)]

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(stochastic_execution, args)

    save_results(backend, results)
    return results


def stochastic_execution(args: tuple[Backend, tuple[QuantumCircuit, NoiseModel | None]]) -> NDArray | list[int]:
    backend, pack = args
    if isinstance(pack, NoisyCircuitFactory):
        circuit = pack.generate_circuit()
        vector_data = backend.execute(circuit)
    else:
        circuit, noise_model = pack
        vector_data = backend.execute(circuit, noise_model)

    return vector_data if backend.full_state_memory else measure_state(vector_data)

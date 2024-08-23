from __future__ import annotations

import getpass
import uuid
from pathlib import Path

import h5py
import numpy as np


def save_full_states(list_of_vectors: list[np.ndarray],
                     file_path: str | Path | None = None, file_name: str | None = None) -> None:
    if file_name is None:
        file_name = "experiment_states.h5"

    if file_path is None:
        username = getpass.getuser()
        file_path = Path(f"/home/{username}/Documents")
    else:
        file_path = Path(file_path)

    full_path = file_path / file_name
    size = list_of_vectors[0].shape

    # Generate random unique names for each vector
    vector_names = [str(uuid.uuid4()) for _ in range(len(list_of_vectors))]

    # Combine names and vectors into a list of dictionaries
    list_of_vectors = [{"name": name, "data": vector} for name, vector in zip(vector_names, list_of_vectors)]

    # Open the HDF5 file in write mode
    with h5py.File(full_path, "w") as hdf_file:
        # Create a table dataset within the file to store the vectors
        dtype = [("name", "S36"), ("vector_data", np.complex128, size)]
        table_data = np.array(
                [(vector_info["name"].encode("utf-8"), vector_info["data"]) for vector_info in list_of_vectors],
                dtype=dtype
        )

        hdf_file.create_dataset("vectors", data=table_data)

    print(f"States saved to {full_path}")


def save_shots(shots, file_path: str | Path | None = None,
               file_name: str | None = "simulation_results.h5") -> None:
    if file_path is None:
        username = getpass.getuser()
        file_path = Path(f"/home/{username}/Documents")
    else:
        file_path = Path(file_path)

    full_path = file_path / file_name
    indexes = list(range(len(shots)))

    # Combine names and vectors into a list of dictionaries
    list_of_outcomes = [{"shot_nr": nr, "outcome": outcome} for nr, outcome in zip(indexes, shots)]

    with h5py.File(full_path, "w") as hdf_file:
        # Create a table dataset within the file to store the vectors
        dtype = [("nr", int), ("shot", int)]
        table_data = np.array(
                [(shot_info["shot_nr"], shot_info["outcome"]) for shot_info in list_of_outcomes], dtype=dtype
        )
        hdf_file.create_dataset("shots", data=table_data)

    print(f"Simulation results saved to {full_path}")

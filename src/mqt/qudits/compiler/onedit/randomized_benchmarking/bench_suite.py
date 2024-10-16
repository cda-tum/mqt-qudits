import os
import pickle
import numpy as np
import itertools

from mqt.qudits.quantum_circuit import QuantumCircuit


# Define H and S gates for a specific qudit dimension
def get_h_gate(dim):
    circuit_d = QuantumCircuit(1, [dim], 0)
    h = circuit_d.h(0)
    return h.to_matrix()


def get_s_gate(dim):
    circuit_d = QuantumCircuit(1, [dim], 0)
    s = circuit_d.s(0)
    return s.to_matrix()


def matrix_hash(matrix):
    """ Hash a numpy matrix using its byte representation. """
    return hash(matrix.tobytes())


def generate_clifford_group(d, max_length=5):
    # Initialize H and S gates
    h_gate = get_h_gate(d)
    s_gate = get_s_gate(d)

    gates = {'h': h_gate, 's': s_gate}
    clifford_group = {}
    hash_table = set()  # To store matrix hashes for fast lookups

    # Iterate over different combination lengths
    for length in range(1, max_length + 1):
        for seq in itertools.product('hs', repeat=length):
            seq_str = ''.join(seq)
            gate_product = np.eye(d)

            # Multiply gates in the sequence
            for gate in seq:
                gate_product = np.dot(gates[gate], gate_product)

            # Hash the matrix
            matrix_h = matrix_hash(gate_product)

            # Check if this matrix is already in the group using the hash table
            if matrix_h not in hash_table:
                clifford_group[seq_str] = gate_product
                hash_table.add(matrix_h)

    return clifford_group


def get_package_data_path(filename):
    """ Get the relative path to the data directory within the package. """
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return os.path.join(data_dir, filename)


def save_clifford_group_to_file(clifford_group, filename):
    """ Save the Clifford group to the 'data' directory in the current package. """
    filepath = get_package_data_path(filename)
    with open(filepath, 'wb') as f:
        pickle.dump(clifford_group, f)


def load_clifford_group_from_file(filename):
    """ Load the Clifford group from the 'data' directory in the current package. """
    filepath = get_package_data_path(filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes


class Lanes:
    def __init__(self, instructions):
        self.instructions = instructions
        self.index_dict = self.create_lanes()

    def create_lanes(self):
        self.index_dict = {}
        for gate in self.instructions:
            if gate.gate_type == GateTypes.SINGLE:
                index = gate.qudit_index
                if index not in self.index_dict:
                    self.index_dict[index] = []
                self.index_dict[index].append(gate)
            elif gate.gate_type == GateTypes.TWO or gate.gate_type == GateTypes.MULTI:
                indices = gate.qudit_index
                for index in indices:
                    if index not in self.index_dict:
                        self.index_dict[index] = []
                    self.index_dict[index].append(gate)
        return self.index_dict

    def extract_circuit(self):
        combined_list = []
        seen_ids = set()

        for line in sorted(self.index_dict.keys()):  # Iterate over keys in sorted order
            for gate in self.index_dict[line]:
                obj_id = id(gate)
                if obj_id not in seen_ids:
                    combined_list.append(gate)
                    seen_ids.add(obj_id)

        self.instructions = combined_list
        return combined_list

    def replace_gates_in_lane(self, line, start_index, end_index, new_gate):
        # Find the list associated with the line
        if line in self.index_dict:
            gates_of_line = self.index_dict[line]
        else:
            return  # Exit if line not found in index_dict

        # Remove objects within the specified interval [start_index, end_index]
        objects_to_remove = []
        for i in range(start_index, end_index + 1):
            if i < len(gates_of_line):
                objects_to_remove.append(gates_of_line[i])

        for obj in objects_to_remove:
            gates_of_line.remove(obj)

        # Add new_gate at the start_index
        gates_of_line.insert(start_index, new_gate)

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from ..quantum_circuit.gates.virt_rz import VirtRz

if TYPE_CHECKING:
    from ..quantum_circuit import QuantumCircuit
    from ..quantum_circuit.gate import Gate


class LevelGraph(nx.Graph):
    def __init__(
            self,
            edges: list[tuple[int, int, dict]],
            nodes: list[int],
            nodes_physical_mapping: list[int] | None = None,
            initialization_nodes: list[int] | None = None,
            qudit_index: int | None = None,
            og_circuit: QuantumCircuit = None
    ) -> None:
        super().__init__()
        self.og_circuit: QuantumCircuit = og_circuit
        self.qudit_index: int = qudit_index
        self.logic_nodes: list[int] = nodes
        self.add_nodes_from(self.logic_nodes)

        if nodes_physical_mapping:
            self.logic_physical_map(nodes_physical_mapping)

        self.add_edges_from(edges)

        if initialization_nodes:
            inreach_nodes: list[int] = [x for x in nodes if x not in initialization_nodes]
            self.define__states(initialization_nodes, inreach_nodes)

    def phase_storing_setup(self) -> None:
        for node in self.nodes:
            node_dict = self.nodes[node]
            if "phase_storage" not in node_dict:
                node_dict["phase_storage"] = 0

    def distance_nodes(self, source: int, target: int) -> int:
        path = nx.shortest_path(self, source, target)
        return len(path) - 1

    def distance_nodes_pi_pulses_fixed_ancilla(self, source: int, target: int) -> int:
        path = nx.shortest_path(self, source, target)
        negs = 0
        pos = 0
        for n in path:
            if n >= 0:
                pos += 1
            else:
                negs += 1
        return (2 * negs) - 1 + (pos) - 1

    def logic_physical_map(self, physical_nodes: list[int]) -> None:
        logic_phy_map = dict(zip(self.logic_nodes, physical_nodes))
        nx.set_node_attributes(self, logic_phy_map, "lpmap")

    def define__states(self, initialization_nodes: list[int], inreach_nodes: list[int]) -> None:
        inreach_dictionary = dict.fromkeys(inreach_nodes, "r")
        initialization_dictionary = dict.fromkeys(initialization_nodes, "i")

        for _n in inreach_dictionary:
            nx.set_node_attributes(self, inreach_dictionary, name="level")

        for _n in initialization_dictionary:
            nx.set_node_attributes(self, initialization_dictionary, name="level")

    @staticmethod
    def update_list(lst_: list[tuple[int, int]], num_a: int, num_b: int) -> list[tuple[int, int]]:
        new_lst = []

        mod_index = []
        for t in lst_:
            tupla = [0, 0]
            if t[0] == num_a:
                tupla[0] = 1
            elif t[0] == num_b:
                tupla[0] = 2

            if t[1] == num_a:
                tupla[1] = 1
            elif t[1] == num_b:
                tupla[1] = 2

            mod_index.append(tupla)

        for i, t in enumerate(lst_):
            substituter = list(t)

            if mod_index[i][0] == 1:
                substituter[0] = num_b
            elif mod_index[i][0] == 2:
                substituter[0] = num_a

            if mod_index[i][1] == 1:
                substituter[1] = num_b
            elif mod_index[i][1] == 2:
                substituter[1] = num_a

            new_lst.append(tuple(substituter))

        return new_lst

    @staticmethod
    def deep_copy_func(l_n: list[tuple[int, int, dict]] | list[tuple[int, dict]]) -> list[tuple[int, int, dict]] | list[tuple[int, dict]]:
        cpy_list = []
        for li in l_n:
            d2 = copy.deepcopy(li)
            cpy_list.append(d2)

        return cpy_list

    @staticmethod
    def index(lev_graph: list[tuple[int, dict]], node: int) -> int | None:
        for i in range(len(lev_graph)):
            if lev_graph[i][0] == node:
                return i
        return None

    def swap_node_attributes(self, node_a: int, node_b: int) -> list[tuple[int, dict]]:
        nodelistcopy = self.deep_copy_func(list(self.nodes(data=True)))
        node_a = self.index(nodelistcopy, node_a)
        node_b = self.index(nodelistcopy, node_b)

        dict_attr_inode = nodelistcopy[0][1]
        for attr in list(dict_attr_inode.keys()):
            attr_a = nodelistcopy[node_a][1][attr]
            attr_b = nodelistcopy[node_b][1][attr]
            nodelistcopy[node_a][1][attr] = attr_b
            nodelistcopy[node_b][1][attr] = attr_a

        return nodelistcopy

    def swap_node_attr_simple(self, node_a: int, node_b: int) -> None:
        res_list = [x[0] for x in self.nodes(data=True)]
        node_a = res_list.index(node_a)
        node_b = res_list.index(node_b)

        inode = self.fst_inode
        if "phase_storage" in self.nodes[inode]:
            phi_a = self.nodes[node_a]["phase_storage"]
            phi_b = self.nodes[node_b]["phase_storage"]
            self.nodes[node_a]["phase_storage"] = phi_b
            self.nodes[node_b]["phase_storage"] = phi_a

    def swap_nodes(self, node_a: int, node_b: int) -> LevelGraph:
        nodes = self.swap_node_attributes(node_a, node_b)
        # ------------------------------------------------
        new_graph = LevelGraph([], nodes)

        edges = self.deep_copy_func(list(self.edges))

        attribute_list = [self.get_edge_data(*e).copy() for e in edges]

        swapped_nodes_edges = self.update_list(edges, node_a, node_b)

        new_edge_list = []
        for i, e in enumerate(swapped_nodes_edges):
            new_edge_list.append((*e, attribute_list[i]))

        new_graph.add_edges_from(new_edge_list)

        return new_graph

    def get_vrz_gates(self) -> list[Gate]:
        matrices = []
        for node in self.nodes:
            node_dict = self.nodes[node]
            if ("phase_storage" in node_dict and (node_dict["phase_storage"] > 1e-3
                                                  or np.mod(node_dict["phase_storage"], 2 * np.pi) > 1e-3)):
                phy_n_i = self.nodes[node]["lpmap"]

                # phase_gate = VirtRz(node_dict["phase_storage"], phy_n_i, len(list(self.nodes)))
                phase_gate = VirtRz(
                        self.og_circuit,
                        "VirtRz_egraph",
                        self.qudit_index,
                        [phy_n_i, node_dict["phase_storage"]],
                        self.og_circuit.dimensions[self.qudit_index],
                )
                matrices.append(phase_gate)

        return matrices

    def get_node_sensitivity_cost(self, node: int) -> float | int:
        neighbs = list(self.neighbors(node))

        total_sensibility = 0
        for i in range(len(neighbs)):
            total_sensibility += self[node][neighbs[i]]["sensitivity"]

        return total_sensibility

    def get_edge_sensitivity(self, node_a: int, node_b: int) -> float | int:
        return self[node_a][node_b]["sensitivity"]

    @property
    def fst_rnode(self) -> int:
        r_node = [x for x, y in self.nodes(data=True) if y["level"] == "r"]
        return r_node[0]

    @property
    def fst_inode(self) -> int:
        inode = [x for x, y in self.nodes(data=True) if y["level"] == "i"]
        return inode[0]

    def is_irnode(self, node: int) -> bool:
        irnodes = [x for x, y in self.nodes(data=True) if y["level"] == "r"]
        return node in irnodes

    def is_inode(self, node: int) -> bool:
        inodes = [x for x, y in self.nodes(data=True) if y["level"] == "i"]
        return node in inodes

    @property
    def log_phy_map(self) -> list[int]:
        nodes = self.nodes

        return [N[1]["lpmap"] for key in nodes for N in self.nodes(data=True) if N[0] == key]

    def __str__(self) -> str:
        return str(self.nodes(data=True)) + "\n" + str(self.edges(data=True))

    def set_circuit(self, circuit: QuantumCircuit) -> None:
        self.og_circuit = circuit

    def set_qudits_index(self, index: int) -> None:
        self.qudit_index = index

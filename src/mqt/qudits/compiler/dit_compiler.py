from __future__ import annotations

import typing
from typing import Optional

from ..core.custom_python_utils import append_to_front
from ..quantum_circuit.components.extensions.gate_types import GateTypes
from . import CompilerPass
from .naive_local_resynth import NaiveLocResynthOptPass
from .onedit import LogLocQRPass, PhyLocAdaPass, PhyLocQRPass, ZPropagationOptPass, ZRemovalOptPass
from .twodit import LogEntQRCEXPass
from .twodit.entanglement_qr.phy_ent_qr_cex_decomp import PhyEntQRCEXPass

if typing.TYPE_CHECKING:
    from ..quantum_circuit import QuantumCircuit
    from ..quantum_circuit.gate import Gate
    from ..simulation.backends.backendv2 import Backend


class QuditCompiler:
    passes_enabled: typing.ClassVar = {
        "PhyLocQRPass":           PhyLocQRPass,
        "PhyLocAdaPass":          PhyLocAdaPass,
        "LocQRPass":              PhyLocQRPass,
        "LocAdaPass":             PhyLocAdaPass,
        "LogLocQRPass":           LogLocQRPass,
        "ZPropagationOptPass":    ZPropagationOptPass,
        "ZRemovalOptPass":        ZRemovalOptPass,
        "LogEntQRCEXPass":        LogEntQRCEXPass,
        "PhyEntQRCEXPass":        PhyEntQRCEXPass,
        "NaiveLocResynthOptPass": NaiveLocResynthOptPass,
    }

    def __init__(self) -> None:
        pass

    def compile(self, backend: Backend, circuit: QuantumCircuit, passes_names: list[str]) -> QuantumCircuit:
        """ Method compiles with passes chosen"""
        passes_dict = {}
        new_instr = []
        # Instantiate and execute created classes
        for compiler_pass_name in passes_names:
            compiler_pass = self.passes_enabled[compiler_pass_name]
            decomposition = compiler_pass(backend)
            if "Loc" in str(compiler_pass):
                passes_dict[GateTypes.SINGLE] = decomposition
            elif "Ent" in str(compiler_pass):
                passes_dict[GateTypes.TWO] = decomposition
            elif "Multi" in str(compiler_pass):
                passes_dict[GateTypes.MULTI] = decomposition

        for gate in reversed(circuit.instructions):
            decomposer = typing.cast(Optional[CompilerPass], passes_dict.get(gate.gate_type))
            if decomposer is not None:
                new_instructions = decomposer.transpile_gate(gate)
                append_to_front(new_instr, new_instructions)
                # new_instr.extend(new_instructions)
            else:
                append_to_front(new_instr, gate)
                #new_instr.append(gate)

        transpiled_circuit = circuit.copy()
        mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_mapping(mappings)
        return transpiled_circuit.set_instructions(new_instr)

    def compile_O0(self, backend: Backend, circuit: QuantumCircuit) -> QuantumCircuit:  # noqa: N802
        """ Method compiles with PHY LOC QR and PHY ENT QR CEX with no optimization"""
        passes = ["PhyLocQRPass", "PhyEntQRCEXPass"]
        compiled = self.compile(backend, circuit, passes)

        mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        compiled.set_mapping(mappings)
        return compiled

    @staticmethod
    def compile_O1_resynth(backend: Backend, circuit: QuantumCircuit) -> QuantumCircuit:  # noqa: N802
        """ Method compiles with PHY LOC QR and PHY ENT QR CEX with a resynth steps """
        phyloc = PhyLocQRPass(backend)
        phyent = PhyEntQRCEXPass(backend)
        resynth = NaiveLocResynthOptPass(backend)

        circuit = resynth.transpile(circuit)
        new_instructions = []
        for gate in reversed(circuit.instructions):
            ins: list[Gate] = []
            if gate.gate_type is GateTypes.SINGLE:
                ins = phyloc.transpile_gate(gate)
                append_to_front(new_instructions, ins)
            else:
                ins = phyent.transpile_gate(gate)
                append_to_front(new_instructions, ins)

        transpiled_circuit = circuit.copy()
        mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_mapping(mappings)
        return transpiled_circuit.set_instructions(new_instructions)

    @staticmethod
    def compile_O1_adaptive(backend: Backend, circuit: QuantumCircuit) -> QuantumCircuit:  # noqa: N802
        """ Method compiles with PHY LOC ADA and PHY ENT QR CEX"""
        phyent = PhyEntQRCEXPass(backend)
        phyloc = PhyLocAdaPass(backend)
        new_instructions = []
        for gate in reversed(circuit.instructions):
            ins: list[Gate] = []
            if gate.gate_type is GateTypes.SINGLE:
                ins = phyloc.transpile_gate(gate)
                append_to_front(new_instructions, ins)
            else:
                ins = phyent.transpile_gate(gate)
                append_to_front(new_instructions, ins)

        transpiled_circuit = circuit.copy()
        transpiled_circuit.set_instructions(new_instructions)
        mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_mapping(mappings)

        z_propagation_pass = ZPropagationOptPass(backend=backend, back=False)
        new_transpiled_circuit = z_propagation_pass.transpile(transpiled_circuit)

        return new_transpiled_circuit

    @staticmethod
    def compile_O2(backend: Backend, circuit: QuantumCircuit) -> QuantumCircuit:  # noqa: N802
        """ Method compiles with PHY LOC ADA and PHY ENT QR CEX with a resynth steps """
        phyloc = PhyLocAdaPass(backend)
        phyent = PhyEntQRCEXPass(backend)
        resynth = NaiveLocResynthOptPass(backend)

        circuit = resynth.transpile(circuit)
        new_instructions = []
        for gate in reversed(circuit.instructions):
            ins: list[Gate] = []
            if gate.gate_type is GateTypes.SINGLE:
                ins = phyloc.transpile_gate(gate)
                append_to_front(new_instructions, ins)
            else:
                ins = phyent.transpile_gate(gate)
                append_to_front(new_instructions, ins)

        transpiled_circuit = circuit.copy()
        mappings = []
        for i, graph in enumerate(backend.energy_level_graphs):
            if i < circuit.num_qudits:
                mappings.append([lev for lev in graph.log_phy_map if lev < circuit.dimensions[i]])
        transpiled_circuit.set_mapping(mappings)
        return transpiled_circuit.set_instructions(new_instructions)

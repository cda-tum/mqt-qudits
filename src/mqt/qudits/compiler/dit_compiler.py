from __future__ import annotations

import numpy as np

from .naive_local_resynth import NaiveLocResynthOptPass
from .onedit import LogLocQRPass, PhyLocAdaPass, PhyLocQRPass, ZPropagationOptPass, ZRemovalOptPass
from .twodit import LogEntQRCEXPass
from .twodit.entanglement_qr.phy_ent_qr_cex_decomp import PhyEntQRCEXPass
from ..core.lanes import Lanes
from ..quantum_circuit.components.extensions.gate_types import GateTypes


class QuditCompiler:
    passes_enabled = {
        "PhyLocQRPass": PhyLocQRPass,
        "PhyLocAdaPass": PhyLocAdaPass,
        "LocQRPass": PhyLocQRPass,
        "LocAdaPass": PhyLocAdaPass,
        "LogLocQRPass": LogLocQRPass,
        "ZPropagationOptPass": ZPropagationOptPass,
        "ZRemovalOptPass": ZRemovalOptPass,
        "LogEntQRCEXPass": LogEntQRCEXPass,
        "PhyEntQRCEXPass": PhyEntQRCEXPass,
        "NaiveLocResynthOptPass": NaiveLocResynthOptPass
    }

    def __init__(self) -> None:
        pass

    def compile(self, backend, circuit, passes_names):
        passes_dict = {}
        new_instr = []
        # Instantiate and execute created classes
        for compiler_pass in passes_names:
            compiler_pass = self.passes_enabled[compiler_pass]
            decomposition = compiler_pass(backend)
            if "Loc" in compiler_pass:
                passes_dict[GateTypes.SINGLE] = decomposition
            elif "Ent" in compiler_pass:
                passes_dict[GateTypes.TWO] = decomposition
            elif "Multi" in compiler_pass:
                passes_dict[GateTypes.MULTI] = decomposition
        for gate in circuit.instructins:
            new_instr.append(passes_dict[gate.gate_type](gate))

        circuit.set_instruct(new_instr)
        return circuit

    def compile_O0(self, backend, circuit):
        passes = ["PhyLocQRPass", "PhyEntQRCEXPass"]
        return self.compile(backend, circuit, passes)

    def compile_O1(self, backend, circuit):
        phyloc = PhyLocAdaPass(backend)
        phyent = PhyEntQRCEXPass(backend)
        resynth = NaiveLocResynthOptPass(backend)
        circuit = resynth.transpile(circuit)

        lanes = Lanes(circuit)
        new_instructions = []
        for gate in circuit.instructions:
            if gate.gate_type is GateTypes.SINGLE:
                new_instructions += phyloc.traspile_gate(gate, lanes.next_is_local(gate))
            else:
                new_instructions += phyent.traspile_gate(gate)

        transpiled_circuit = circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)

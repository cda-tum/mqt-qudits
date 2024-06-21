from __future__ import annotations

from .onedit import LogLocQRPass, PhyLocAdaPass, PhyLocQRPass, ZPropagationPass, ZRemovalPass
from .twodit import LogEntQRCEXPass
from .twodit.entanglement_qr.phy_ent_qr_cex_decomp import PhyEntQRCEXPass


class QuditCompiler:
    passes_enabled = {
        "PhyLocQRPass": PhyLocQRPass,
        "PhyLocAdaPass": PhyLocAdaPass,
        "LocQRPass": PhyLocQRPass,
        "LocAdaPass": PhyLocAdaPass,
        "LogLocQRPass": LogLocQRPass,
        "ZPropagationPass": ZPropagationPass,
        "ZRemovalPass": ZRemovalPass,
        "LogEntQRCEXPass": LogEntQRCEXPass,
        "PhyEntQRCEXPass": PhyEntQRCEXPass
    }

    def __init__(self) -> None:
        pass

    def compile(self, backend, circuit, passes_names):
        # Instantiate and execute created classes
        for compiler_pass in passes_names:
            compiler_pass = self.passes_enabled[compiler_pass]
            decomposition = compiler_pass(backend)
            circuit = decomposition.transpile(circuit)
        return circuit

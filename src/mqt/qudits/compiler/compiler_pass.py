from __future__ import annotations

from abc import ABC, abstractmethod


class CompilerPass(ABC):
    def __init__(self, backend, **kwargs) -> None:
        self.backend = backend

    def transpile_gate(self, gate) -> None:
        pass

    @abstractmethod
    def transpile(self, circuit):
        pass

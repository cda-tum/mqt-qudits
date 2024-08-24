from __future__ import annotations


class NodeNotFoundException(ValueError):
    def __init__(self, value) -> None:
        self.value = value

    def __str__(self) -> str:
        return repr(self.value)


class SequenceFoundException(RuntimeError):
    def __init__(self, node_key: int = -1) -> None:
        self.last_node_id = node_key

    def __str__(self) -> str:
        return repr(self.last_node_id)


class RoutingException(RuntimeError):
    def __init__(self) -> None:
        self.message = "ROUTING PROBLEM STUCK!"

    def __str__(self) -> str:
        return repr(self.message)


class FidelityReachException(ValueError):
    def __init__(self, message: str = "") -> None:
        self.message = message

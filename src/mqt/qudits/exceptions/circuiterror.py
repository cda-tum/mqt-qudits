from __future__ import annotations


class CircuitError(Exception):
    def __init__(self, message: str) -> None:
        print(message)


class InvalidQuditDimensionError(ValueError):
    """Raised when the qudit dimension is invalid for the S gate."""

    def __init__(self, message: str) -> None:
        print(message)

class ShapeMismatchError(ValueError):
    """Raised when input arrays have mismatched shapes."""

    def __init__(self, message: str) -> None:
        print(message)
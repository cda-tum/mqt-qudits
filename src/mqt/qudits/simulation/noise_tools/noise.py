from __future__ import annotations


class Noise:
    """Represents a noise model with depolarizing and dephasing probabilities."""

    probability_depolarizing: float
    probability_dephasing: float


class SubspaceNoise:
    """Represents physial noises for each level transitions."""

    probs: dict[tuple[int, int], Noise]

    def __init__(self, new_dict: dict[tuple[int, int], Noise]) -> None:
        self.add_noises(new_dict)

    def __init__(self, probability_depolarizing: float, probability_dephasing: float, lev_a: int, lev_b: int) -> None:
        self.add_noise([lev_a, lev_b], Noise(probability_depolarizing, probability_dephasing))

    def __init__(
        self,
        probability_depolarizing: float,
        probability_dephasing: float,
        levels: tuple[int, int] | list[tuple[int, int]],
    ) -> None:
        if isinstance(levels, tuple[int, int]):
            self.add_noise(levels, Noise(probability_depolarizing, probability_dephasing))
        elif isinstance(levels, list[tuple[int, int]]):
            for lev in levels:
                self.add_noise(lev, Noise(probability_depolarizing, probability_dephasing))
        else:
            raise NotImplementedError

    def add_noise(self, lev_a: int, lev_b: int, noise: Noise) -> None:
        if lev_b < lev_a:
            lev_a, lev_b = lev_b, lev_a
        if lev_a != lev_b:
            msg = "The levels in the subspace noise should be different!"
            raise ValueError(msg)
        if (lev_a, lev_b) not in self.probs:
            msg = "The same level physical noise is defined for multiple times!"
            raise ValueError(msg)
        self.probs[lev_a, lev_b] = noise

    def add_noises(self, noises: dict[tuple[int, int], Noise]) -> None:
        for levs, noise in noises.items():
            self.add_noise(levs, noise)


class NoiseModel:
    """Represents a quantum noise model for various gates and qudit configurations."""

    def __init__(self) -> None:
        self.quantum_errors: dict[str, dict[str, SubspaceNoise]] = {}

    def _add_quantum_error(self, noise: SubspaceNoise | Noise, gates: list[str], mode: str) -> None:
        """Helper method to add quantum errors to the model.

        Args:
            noise (SubspaceNoise | Noise): The subspace noise model to add.
            gates (List[str]): List of gate names to apply the noise to.
            mode (Union[Tuple[int, ...], Literal["local", "all", "nonlocal", "target", "control"]]): The mode or qudit configuration for the noise.
        """
        for gate in gates:
            if gate not in self.quantum_errors:
                self.quantum_errors[gate] = {}
            if mode not in self.quantum_errors[gate]:
                self.quantum_errors[gate][mode] = noise  # empty case
            else:
                if isinstance(noise, Noise):
                    msg = "Mathematical noise is defined for multiple times!"
                    raise ValueError(msg)
                self.quantum_errors[gate][mode].add_noises(
                    noise.probs
                )  # add the noise info to the existing SubspaceNoise instance

    def add_quantum_error_locally(self, noise: Noise | SubspaceNoise, gates: list[str]) -> None:
        """Add a quantum error locally to all qudits for specified gates."""
        self._add_quantum_error(noise, gates, "local")

    def add_all_qudit_quantum_error(self, noise: Noise | SubspaceNoise, gates: list[str]) -> None:
        """Add a quantum error to all qudits for specified gates."""
        self._add_quantum_error(noise, gates, "all")

    def add_nonlocal_quantum_error(self, noise: Noise | SubspaceNoise, gates: list[str]) -> None:
        """Add a nonlocal quantum error for specified gates."""
        self._add_quantum_error(noise, gates, "nonlocal")

    def add_nonlocal_quantum_error_on_target(self, noise: Noise | SubspaceNoise, gates: list[str]) -> None:
        """Add a nonlocal quantum error on target qudits for specified gates."""
        self._add_quantum_error(noise, gates, "target")

    def add_nonlocal_quantum_error_on_control(self, noise: Noise | SubspaceNoise, gates: list[str]) -> None:
        """Add a nonlocal quantum error on control qudits for specified gates."""
        self._add_quantum_error(noise, gates, "control")

    @property
    def basis_gates(self) -> list[str]:
        """Get the list of basis gates in the noise model."""
        return list(self.quantum_errors.keys())

    def __str__(self) -> str:
        """Return a string representation of the NoiseModel."""
        info_str = "NoiseModel Info:\n"
        for gate, errors in self.quantum_errors.items():
            for mode, subnoise in errors.items():
                info_str += f"Gate: {gate}, Mode: {mode}, SubspaceNoise: {subnoise}\n"
        return info_str

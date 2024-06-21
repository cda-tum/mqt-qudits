from mqt.qudits.compiler import CompilerPass


class NaiveLocResynthPass(CompilerPass):
    def __init__(self, backend) -> None:
        super().__init__(backend)

    def transpile(self, circuit):
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions = []

        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)

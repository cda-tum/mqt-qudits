from mqt.interface.qasm import QASM

qasm = """
        DITQASM 2.0;
        qreg q [3][3,2,3];
        qreg j [2][2,2];
        creg meas[3];
        h q[2];
        cx q[2],q[1];
        cx j[1],q[0];
        rxy ( pi, pi/2 , 0, 1) q[0];
        barrier q[0],q[1],q[2];
        measure q[0] -> meas[0];
        measure q[1] -> meas[1];
        measure q[2] -> meas[2];
        """

print(QASM().parse_ditqasm2_str(qasm))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

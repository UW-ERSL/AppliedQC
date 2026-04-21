import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit.circuit.library import GroverOperator

def build_grover_operator(A_circuit, good_state_qubits):
    """
    Build the Grover operator Q from circuit A.
    good_state_qubits: indices of qubits that define the good subspace
    """
    grover_op = GroverOperator(oracle=A_circuit,
                               reflection_qubits=good_state_qubits)
    return grover_op

def myQAE(A_circuit, good_state_qubits, m, nShots=10000):
    """
    Quantum Amplitude Estimation.
    A_circuit   : QuantumCircuit preparing the state
    good_state_qubits: list of qubit indices defining the good subspace
    m           : number of precision qubits
    Returns     : estimated amplitude a_tilde
    """
    n = A_circuit.num_qubits
    prec_reg = QuantumRegister(m, 'prec')
    sys_reg  = QuantumRegister(n, 'sys')
    c_reg    = ClassicalRegister(m, 'c')
    qc = QuantumCircuit(prec_reg, sys_reg, c_reg)

    # Step 1: State preparation
    qc.append(A_circuit, sys_reg)

    # Step 2: Hadamard on precision register
    for j in range(m):
        qc.h(prec_reg[j])

    # Step 3: Controlled Q^{2^j}
    Q = build_grover_operator(A_circuit, good_state_qubits)
    for j in range(m):
        power = 2**j
        Q_pow = Q.power(power).control(1)
        qc.append(Q_pow, [prec_reg[j]] + list(sys_reg))

    # Step 4: Inverse QFT on precision register
    iqft = QFT(num_qubits=m, inverse=True)
    qc.append(iqft, prec_reg)

    # Step 5: Measure precision register
    qc.measure(prec_reg, c_reg)

    # Execute and decode
    sim = AerSimulator()
    result = sim.run(qc, shots=nShots).result()
    counts = result.get_counts()

    # Decode most likely outcome
    top_bitstring = max(counts, key=counts.get)
    phi_tilde = int(top_bitstring, 2) / 2**m
    theta_tilde = np.pi * phi_tilde
    a_tilde = np.sin(theta_tilde)**2

    return a_tilde, phi_tilde, counts
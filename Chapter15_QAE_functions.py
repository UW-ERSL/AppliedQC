import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit.circuit.library import GroverOperator
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import SparsePauliOp

from Chapter08_QuantumGates_functions import (simulate_statevector, simulate_measurements, runCircuitOnIBMQuantum, 
                                              findActualHardwareRequirements, plot_measurement_results)

from Chapter14_MatrixEncoding_functions import LCU_Ax


def build_observable_circuit(A, x, f):
    f = f / np.linalg.norm(f)

    # Get alpha from Pauli decomposition
    pauli_split = SparsePauliOp.from_operator(A)
    alpha = np.sum(np.abs(pauli_split.coeffs))
    num_system = int(np.ceil(np.log2(A.shape[0])))

    # Post-selected system state: A|x> / ||A|x>||
    Ax = A @ x
    Ax_norm = Ax / np.linalg.norm(Ax)

    # System-only circuit — no ancilla
    qc = QuantumCircuit(num_system)

    # Step 1: Prepare A|x>/||A|x>||
    qc.append(StatePreparation(Ax_norm.astype(complex), label='A|x>'), range(num_system))

    # Step 2: U_f^dag — rotates |f> -> |0>
    qc.append(StatePreparation(f.astype(complex), label='Uf').inverse(), range(num_system))

    # Step 3: X gates — IQAE's good state is |1...1>, ours is |0...0>
    for i in range(num_system):
        qc.x(i)

    p_success = np.linalg.norm(Ax)**2 / alpha**2
    metadata = {
        'alpha': alpha,
        'num_system': num_system,
        'num_ancilla': 0,
        'p_success': p_success,
        'good_qubits': list(range(num_system)),
    }
    return qc, metadata

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
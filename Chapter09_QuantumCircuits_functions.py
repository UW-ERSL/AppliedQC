
"""
Quantum Circuits for Woodbury Algorithm
========================================
Hadamard test and inner product estimation for quantum linear algebra.

References:
- O'Malley et al. (2024): "Quantum Woodbury algorithm", Quantum 8, 1188
- Nielsen & Chuang (2010): Chapter on quantum algorithms
"""

import numpy as np
from collections import defaultdict
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, StatevectorSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from Chapter08_QuantumGates_functions import simulate_measurements #type: ignore


def ryMatrix(alpha):
	return np.array([[np.cos(alpha/2), -np.sin(alpha/2)], [np.sin(alpha/2), np.cos(alpha/2)]])

def rzMatrix(omega):
	return np.array([[np.exp(-1j*omega/2), 0], [0, np.exp(1j*omega/2)]])

# State preparation circuits (Hadamard = uniform superposition)
def hadamards(n_qubits):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    return qc


def hadamard_test_circuit(u, psi_prep, complex_test=False):
    """
    Hadamard test circuit for estimating <psi|U|psi>.
    
    Parameters
    ----------
    u : Gate
        Unitary operator to test
    psi_prep : QuantumCircuit  
        Circuit preparing |psi>
    complex_test : bool
        False for Re(<psi|U|psi>), True for Im(<psi|U|psi>)
    
    Returns
    -------
    QuantumCircuit
        Hadamard test circuit with measurement on ancilla qubit
    """
    u_controlled = u.control(1)
    ht_circuit = QuantumCircuit(u_controlled.num_qubits, 1)
    ht_circuit.h(0)
    if complex_test:
        ht_circuit.p(-np.pi / 2, 0)
    ht_circuit.append(psi_prep, list(range(1, u_controlled.num_qubits)))
    ht_circuit.append(u_controlled, list(range(u_controlled.num_qubits)))
    ht_circuit.h(0)
    ht_circuit.measure(0, 0)
    return ht_circuit


def hadamard_test_expval(counts, bias_matrix=None):
    """
    Extract expectation value from Hadamard test counts.
    
    Parameters
    ----------
    counts : dict
        {'0': n0, '1': n1} measurement counts
    bias_matrix : array, optional
        2x2 measurement error correction matrix
    
    Returns
    -------
    float
        Expectation value (2*P(0) - 1)
    """
    c0 = counts.get('0', 0)
    c1 = counts.get('1', 0)
    
    if bias_matrix is not None:
        c0, c1 = np.linalg.solve(bias_matrix, [c0, c1])
    
    total = c0 + c1
    return (c0 - c1) / total if total > 0 else 0.0


def hadamard_inner_product(y_prep, x_prep, shots, backend=None, isreal=False):
    """
    Compute <y|x> using Hadamard test.
    """
    if backend is None:
        backend = Aer.get_backend('qasm_simulator')
    
    # Build U = |x><y|^† circuit - CONVERT TO GATES FIRST
    qc = QuantumCircuit(x_prep.num_qubits)
    qc.append(x_prep.to_gate(), range(x_prep.num_qubits))
    qc.append(y_prep.inverse().to_gate(), range(x_prep.num_qubits))
    u_gate = qc.to_gate()
    
    # Hadamard tests for real and imaginary parts
    ht_real = hadamard_test_circuit(u_gate, y_prep.to_gate(), complex_test=False)
    
    if isreal:
        counts_real = simulate_measurements(ht_real, shots= shots)
        return hadamard_test_expval(counts_real)
    else:
        ht_imag = hadamard_test_circuit(u_gate, y_prep.to_gate(), complex_test=True)
        counts_real = simulate_measurements(ht_real, shots= shots)
        counts_imag = simulate_measurements(ht_imag, shots= shots)
        
        real_part = hadamard_test_expval(counts_real)
        imag_part = hadamard_test_expval(counts_imag)
        return complex(real_part, imag_part)


def woodbury_rank1_query(z_prep, b_prep, v_prep, u_prep, alpha, beta, shots, backend=None):
    """
    Quantum Woodbury algorithm for rank-1 update.
    
    Computes: <z|x> where x solves (K + alpha*beta*u*v^T)x = b
    Using Woodbury: x = b - (alpha*beta*<v|b>)/(1 + alpha*beta*<v|u>) * u
    
    Parameters
    ----------
    z_prep : QuantumCircuit
        Query vector |z>
    b_prep : QuantumCircuit
        Right-hand side |b>
    v_prep, u_prep : QuantumCircuit
        Update vectors |v>, |u>
    alpha, beta : float
        Woodbury parameters
    shots : int
        Measurement shots per Hadamard test
    backend : Backend, optional
        Quantum backend
    
    Returns
    -------
    float
        Query result <z|x>
    """
    # Four inner products via Hadamard tests
    zb = hadamard_inner_product(z_prep, b_prep, shots, backend, isreal=True)
    vb = hadamard_inner_product(v_prep, b_prep, shots, backend, isreal=True)
    vu = hadamard_inner_product(v_prep, u_prep, shots, backend, isreal=True)
    zu = hadamard_inner_product(z_prep, u_prep, shots, backend, isreal=True)
    
    # Woodbury formula
    return zb - alpha * beta * vb / (1 + alpha * beta * vu) * zu

def inner_product_estimation(se_unit_vector, test_unit_vector, shots = 10000):
    """
    Estimate |<s|x>|^2 using Hadamard test.
    
    Parameters
    ----------
    se_unit_vector : array
        State |s> vector
    test_unit_vector : array
        Test |x> bitstring
    shots : int
        Measurement shots
    backend : Backend, optional
        Quantum backend
    
    Returns
    -------
    float
        Estimated |<s|x>|^2
    """
    # --- 2. Circuit Setup ---
    num_data_qubits = int(np.log2(len(se_unit_vector))) # log2(32) = 5
    q_aux = QuantumRegister(1, 'aux')
    q_psi = QuantumRegister(num_data_qubits, 'sens')
    q_phi = QuantumRegister(num_data_qubits, 'design')
    c_res = ClassicalRegister(1, 'm')

     # Build circuit
    qc = QuantumCircuit(q_aux, q_psi, q_phi, c_res)
    qc.initialize(se_unit_vector, q_psi)
    qc.initialize(test_unit_vector, q_phi)

    qc.barrier()

    qc.h(q_aux[0])
    for i in range(num_data_qubits):
        qc.cswap(q_aux[0], q_psi[i], q_phi[i])
    qc.h(q_aux[0])

    qc.measure(q_aux, c_res)
    # Simulate circuit
    counts = simulate_measurements(qc, shots= shots )
    
    # Calculate |<s|x>|^2
    p0 = counts.get('0', 0) / shots
    overlap_squared = max(0, 2 * p0 - 1)
    
    return overlap_squared





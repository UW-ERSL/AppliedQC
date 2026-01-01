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
        counts_real = simulateCircuit(ht_real, shots)
        return hadamard_test_expval(counts_real)
    else:
        ht_imag = hadamard_test_circuit(u_gate, y_prep.to_gate(), complex_test=True)
        counts_real = simulateCircuit(ht_real, shots)
        counts_imag = simulateCircuit(ht_imag, shots)
        
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


def simulateCircuit(circuit, shots=1000):
    """Use existing function from Chapter 6"""
    from Chapter06_QuantumGates_functions import simulateCircuit as sim
    return sim(circuit, shots)

import numpy as np
from qiskit import QuantumCircuit


# Problem: (I + u*v^T)x = b, compute <z|x>
n = 3  # 3 qubits = 8-dimensional vectors

# State preparation circuits (Hadamard = uniform superposition)
def hadamards(n_qubits):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    return qc

z_prep = hadamards(n)  # Query vector |z>
b_prep = hadamards(n)  # RHS |b>
v_prep = hadamards(n)  # Update |v>
u_prep = hadamards(n)  # Update |u>

# Woodbury parameters
alpha = 1.0
beta = 1.0

# Run quantum algorithm
result = woodbury_rank1_query(
    z_prep, b_prep, v_prep, u_prep, 
    alpha, beta, 
    shots=10000
)

print(f"Quantum result: {result:.4f}")
print(f"Expected (analytical): 0.5000")  # For uniform states
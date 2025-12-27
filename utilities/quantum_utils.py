"""
Quantum Computing Utilities
============================

Common helper functions used across multiple chapters of the textbook.

This module provides reusable quantum circuit simulation, measurement,
and analysis utilities for Qiskit-based quantum computing examples.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer


def simulateCircuit(circuit, shots=1000):
    """
    Simulate a quantum circuit using Qiskit Aer simulator.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit to simulate
    shots : int, optional
        Number of measurement shots (default: 1000)
    
    Returns
    -------
    dict
        Dictionary of measurement outcomes and their counts
    
    Example
    -------
    >>> from qiskit import QuantumCircuit
    >>> circuit = QuantumCircuit(1, 1)
    >>> circuit.h(0)
    >>> circuit.measure(0, 0)
    >>> counts = simulateCircuit(circuit, shots=1000)
    >>> print(counts)
    {'0': ~500, '1': ~500}
    """
    backend = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(circuit, backend)
    job = backend.run(transpiled_circuit, shots=shots)
    counts = job.result().get_counts(circuit)
    return counts


def ryMatrix(alpha):
    """
    Create a rotation matrix for R_y(alpha) gate.
    
    Parameters
    ----------
    alpha : float
        Rotation angle in radians
    
    Returns
    -------
    numpy.ndarray
        2x2 rotation matrix
    
    Notes
    -----
    The R_y rotation matrix is:
    [[cos(alpha/2), -sin(alpha/2)],
     [sin(alpha/2),  cos(alpha/2)]]
    """
    return np.array([
        [np.cos(alpha/2), -np.sin(alpha/2)],
        [np.sin(alpha/2),  np.cos(alpha/2)]
    ])


def get_statevector(circuit):
    """
    Get the statevector of a quantum circuit without measurement.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit (should not have measurements)
    
    Returns
    -------
    numpy.ndarray
        Complex statevector array
    
    Example
    -------
    >>> circuit = QuantumCircuit(2)
    >>> circuit.h(0)
    >>> circuit.cx(0, 1)
    >>> sv = get_statevector(circuit)
    >>> print(sv)  # Bell state: |00⟩ + |11⟩ (normalized)
    """
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import Statevector
    
    # Use statevector simulator
    backend = AerSimulator(method='statevector')
    circuit_copy = circuit.copy()
    circuit_copy.save_statevector()
    
    job = backend.run(circuit_copy)
    result = job.result()
    statevector = result.get_statevector()
    
    return np.array(statevector.data)


def print_statevector(statevector, threshold=1e-10, n_qubits=None):
    """
    Pretty print a quantum statevector showing only non-zero amplitudes.
    
    Parameters
    ----------
    statevector : array-like
        Complex amplitudes of quantum state
    threshold : float, optional
        Minimum magnitude to display (default: 1e-10)
    n_qubits : int, optional
        Number of qubits (auto-detected if not provided)
    
    Example
    -------
    >>> sv = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    >>> print_statevector(sv)
    |00⟩: (0.707+0.000j)
    |11⟩: (0.707+0.000j)
    """
    if n_qubits is None:
        n_qubits = int(np.log2(len(statevector)))
    
    print("Statevector:")
    for i, amp in enumerate(statevector):
        if abs(amp) > threshold:
            basis_state = format(i, f'0{n_qubits}b')
            print(f"|{basis_state}⟩: ({amp.real:+.3f}{amp.imag:+.3f}j)")


def measure_expectation(circuit, observable, shots=1000):
    """
    Measure expectation value of an observable (Pauli string).
    
    Parameters
    ----------
    circuit : QuantumCircuit
        Quantum circuit to measure
    observable : str
        Pauli string, e.g., 'ZZ', 'XY', etc.
    shots : int, optional
        Number of measurement shots
    
    Returns
    -------
    float
        Estimated expectation value
    
    Notes
    -----
    This is useful for variational algorithms (VQLS, VQE).
    Requires appropriate measurement basis rotations for X and Y.
    """
    # This is a simplified version - full implementation would handle
    # basis rotations for X and Y measurements
    raise NotImplementedError(
        "Full implementation requires basis change handling. "
        "See VQLS chapter for complete implementation."
    )

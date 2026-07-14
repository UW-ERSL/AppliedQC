"""
Chapter 12: Quantum Vector Encoding - Functions 
"""

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

def gate_count(b, m):
    """
    Count the elementary gates needed to amplitude-encode a state vector.

    Builds an ``m``-qubit circuit that initializes the amplitudes ``b``, transpiles
    it down to the hardware-style basis {u, cx} at maximum optimization, and tallies
    the resulting operations. This quantifies the gate cost of exact vector encoding
    discussed in the chapter.

    Parameters
    ----------
    b : array_like
        State vector of length ``2**m`` giving the amplitudes to encode. Passed to
        ``QuantumCircuit.initialize`` (should be normalized).
    m : int
        Number of qubits in the circuit.

    Returns
    -------
    ops : qiskit.result.Counts or dict
        Mapping from each basis gate name (e.g. ``'u'``, ``'cx'``) to its count in
        the transpiled circuit.
    total : int
        Total number of gates, i.e. the sum of all values in ``ops``.
    """
    qc = QuantumCircuit(m)
    qc.initialize(b)
    qc_decomposed = transpile(qc, basis_gates=['u', 'cx'],
                              optimization_level=3)
    ops = qc_decomposed.count_ops()
    total = sum(ops.values())
    return ops, total
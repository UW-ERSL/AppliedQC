"""
Gate-count helper for Chapter 13, "Vector Encoding".

One function, used to produce Table 13.1: it hands a vector to Qiskit's
general-purpose state-preparation synthesis and counts the resulting gates,
so that the cost of ignoring structure can be compared against the
hand-crafted and PyEncode circuits of the chapter.
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
        ``QuantumCircuit.prepare_state`` (should be normalized).
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
    # prepare_state, not initialize: initialize prepends one reset per qubit,
    # which would add m gates to every count in Table 13.1.
    qc.prepare_state(b, range(m))
    qc_decomposed = transpile(qc, basis_gates=['u', 'cx'],
                              optimization_level=3)
    ops = qc_decomposed.count_ops()
    total = sum(ops.values())
    return ops, total
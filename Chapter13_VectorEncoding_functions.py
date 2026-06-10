"""
Chapter 12: Quantum Vector Encoding - Functions 
"""

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

def gate_count(b, m):
    qc = QuantumCircuit(m)
    qc.initialize(b)
    qc_decomposed = transpile(qc, basis_gates=['u', 'cx'],
                              optimization_level=3)
    ops = qc_decomposed.count_ops()
    total = sum(ops.values())
    return ops, total
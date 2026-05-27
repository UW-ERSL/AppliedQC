"""
Chapter 12: Quantum Vector Encoding - Functions 
"""

import numpy as np
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import  StatePreparation,DiagonalGate
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.library.standard_gates import PhaseGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

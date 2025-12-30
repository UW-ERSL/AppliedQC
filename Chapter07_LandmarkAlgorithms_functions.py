import numpy as np
import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from IPython.display import display
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import UnitaryGate, MCXGate
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit_aer import Aer
from Chapter05_QuantumGates_functions import simulateCircuit #type: ignore
from qiskit.circuit.library import PhaseOracle

def bv_secret_circuit():
    s = '11010' # the hidden string
    n = len(s)
    secretCircuit = QuantumCircuit(n+1)
    for ii, yesno in enumerate(reversed(s)):
        if yesno == '1': 
            secretCircuit.cx(ii+1, 0)
    display(secretCircuit.draw('mpl'))
    U = Operator(secretCircuit)
    return U, n


def grover_secret_circuit():
    expression = '~q_0 & q_1 & (~q_2) & q_3 & q_4'  # secret state 11010
    circuit = PhaseOracle(expression) # convenient way to create oracle
    return circuit

def createPhaseInversionCircuit():
    s = '101'  # secret string
    n = len(s)
    qc = QuantumCircuit(n+1)  # type: ignore
    qr = range(1, n+1)
    anc = 0

    # |-> on ancilla
    qc.x(anc)
    qc.h(anc)

    mx = MCXGate(n, label = '', ctrl_state=s) # type: ignore
    qc.append(mx, list(qr) + [anc])
    display(qc.draw('mpl'))
    return Operator(qc), n # type: ignore


def diffusion_operator(qc, qubits):
    n = len(qubits)
    ancilla = qc.num_qubits - 1  # assume ancilla is last qubit

    # Step 1: Apply H to all data qubits
    for q in qubits:
        qc.h(q)

    # Step 2: Put ancilla in |-> (do once at beginning)
    # qc.x(ancilla); qc.h(ancilla)  # do outside

    # Step 3: Flip ancilla if all qubits are 0
    # → MCX with control on all qubits being 0 → so flip all to expect 1
    for q in qubits:
        qc.x(q)
    qc.append(MCXGate(n), qubits + [ancilla])  # multi-controlled X # type: ignore
    for q in qubits:
        qc.x(q)

    # Step 4: Apply H to all data qubits again
    for q in qubits:
        qc.h(q)
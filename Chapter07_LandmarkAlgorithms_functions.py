"""
Landmark Quantum Algorithms
============================
Implementations of foundational quantum algorithms demonstrating quantum advantage.

Algorithms Included:
--------------------
1. Bernstein-Vazirani (BV): Finds hidden bitstring in single query (vs n queries classically)
2. Grover's Search: Finds marked element in O(sqrt(N)) queries (vs O(N) classically)

Key Quantum Concepts:
---------------------
- Quantum parallelism: Evaluate function on superposition of all inputs simultaneously
- Phase kickback: Oracle marks solutions by phase flip
- Amplitude amplification: Increase probability of measuring correct answer

Computational Complexity:
-------------------------
- BV: 1 quantum query vs n classical queries (exponential speedup)
- Grover: O(sqrt(N)) vs O(N) classical (quadratic speedup)
  Optimal iterations ≈ π/4 * sqrt(N)
  
References:
-----------
- Bernstein & Vazirani (1997): Quantum complexity theory
- Grover (1996): Fast quantum mechanical algorithm for database search
- Nielsen & Chuang (2010): Quantum Computation and Quantum Information, Ch. 6
"""
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
    """
    Bernstein-Vazirani Oracle Circuit
    ==================================
    Creates oracle for BV algorithm that encodes secret bitstring.
    
    Algorithm: Given black-box function f(x) = s·x (mod 2), find secret string s.
    Classical: Requires n queries (one per bit)
    Quantum: Requires only 1 query using superposition
    
    Implementation:
    - Secret string: s = '11010'
    - Oracle applies: |x⟩|y⟩ → |x⟩|y ⊕ (s·x)⟩
    - Using phase kickback with |−⟩ ancilla: |x⟩|−⟩ → (−1)^(s·x)|x⟩|−⟩
    
    Returns:
    --------
    U : Operator
        Unitary operator for oracle
    n : int
        Number of qubits (length of secret string)
    
    Complexity: O(n) gates for n-bit string
    """
    s = '11010'  # The hidden bitstring to find
    n = len(s)
    secretCircuit = QuantumCircuit(n+1)  # n data qubits + 1 ancilla
    
    # Apply CNOT from data qubit i to ancilla if s[i] = '1'
    # This implements f(x) = s·x mod 2
    for ii, yesno in enumerate(reversed(s)):
        if yesno == '1': 
            secretCircuit.cx(ii+1, 0)  # Control on qubit ii+1, target on ancilla 0
    display(secretCircuit.draw('mpl'))
    U = Operator(secretCircuit)
    return U, n


def grover_secret_circuit():
    """
    Grover Oracle using Phase Oracle
    =================================
    Creates oracle for Grover's search algorithm.
    
    Marks target state '11010' by phase flip: |x⟩ → -|x⟩ if x is solution
    Uses Boolean expression: ~q_0 & q_1 & (~q_2) & q_3 & q_4
    This corresponds to bitstring 01010 (reading q_4 q_3 q_2 q_1 q_0)
    
    Grover's Algorithm Complexity:
    - Search space size: N = 2^n
    - Quantum queries: O(sqrt(N)) ≈ π/4 * sqrt(N)
    - Classical queries: O(N) (brute force)
    - Quadratic speedup factor
    
    Returns:
    --------
    circuit : PhaseOracle
        Oracle circuit that marks solution by phase flip
    """
    expression = '~q_0 & q_1 & (~q_2) & q_3 & q_4'  # Target: 11010 (reversed indexing)
    circuit = PhaseOracle(expression)  # Convenient Qiskit method for oracle creation
    return circuit

def createPhaseInversionCircuit():
    """
    Phase Inversion Circuit for Grover's Algorithm
    ===============================================
    Creates circuit that applies phase flip conditioned on secret string.
    
    Technique: Multi-controlled Toffoli with ancilla in |−⟩ state
    - Ancilla starts in |−⟩ = (|0⟩ - |1⟩)/sqrt(2)
    - MCX gate with ctrl_state matching secret flips ancilla
    - Phase kickback: |ψ⟩|−⟩ → -|ψ⟩|−⟩ when control condition met
    
    Parameters:
    -----------
    Secret string: s = '101'
    
    Returns:
    --------
    U : Operator
        Unitary operator for phase inversion
    n : int
        Number of qubits
        
    Note: This is a more explicit construction than PhaseOracle,
    useful for understanding the phase kickback mechanism.
    """
    s = '101'  # Secret string to mark
    n = len(s)
    qc = QuantumCircuit(n+1)  # type: ignore
    qr = range(1, n+1)  # Data qubits
    anc = 0  # Ancilla qubit
    
    # Initialize ancilla to |−⟩ state for phase kickback
    qc.x(anc)
    qc.h(anc)
    
    # Multi-controlled X: Flips ancilla only when control qubits match ctrl_state
    # With ancilla in |−⟩, this applies phase flip to marked state
    mx = MCXGate(n, label = '', ctrl_state=s)  # type: ignore
    qc.append(mx, list(qr) + [anc])
    display(qc.draw('mpl'))
    return Operator(qc), n  # type: ignore


def diffusion_operator(qc, qubits):
    """
    Grover Diffusion Operator (Inversion About Average)
    ===================================================
    Amplifies amplitude of marked states by inverting about average amplitude.
    
    Mathematical Operation: D = 2|ψ⟩⟨ψ| - I
    where |ψ⟩ = (1/sqrt(N)) Σ|x⟩ is equal superposition
    
    Effect: 
    - States with amplitude above average get boosted
    - States below average get suppressed
    - Combined with oracle, amplifies probability of solution
    
    Implementation:
    1. H^⊗n: Transform to computational basis
    2. Conditional phase flip on |00...0⟩ state (using MCX with ancilla)
    3. H^⊗n: Transform back to superposition basis
    
    Result: After k iterations with optimal k ≈ π/4*sqrt(N),
    solution probability approaches 1.
    
    Parameters:
    -----------
    qc : QuantumCircuit
        Circuit to append diffusion operator to
    qubits : list
        Indices of data qubits (excludes ancilla)
        
    Note: Ancilla assumed to be last qubit and already in |−⟩ state
    """
    n = len(qubits)
    ancilla = qc.num_qubits - 1  # Assume ancilla is last qubit

    # Step 1: Apply H to all data qubits (transform to computational basis)
    for q in qubits:
        qc.h(q)

    # Step 2: Ancilla should be in |−⟩ (initialized outside this function)
    # qc.x(ancilla); qc.h(ancilla)  # Done once at circuit start

    # Step 3: Flip ancilla if all qubits are 0 (conditional phase flip)
    # X gates: Transform condition from "all zeros" to "all ones" for MCX
    for q in qubits:
        qc.x(q)
    qc.append(MCXGate(n), qubits + [ancilla])  # Multi-controlled X on all qubits
    for q in qubits:
        qc.x(q)

    # Step 4: Apply H to all data qubits again (transform back to superposition)
    for q in qubits:
        qc.h(q)
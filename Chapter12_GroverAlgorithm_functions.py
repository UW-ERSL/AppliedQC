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
from qiskit import QuantumCircuit

from IPython.display import display
from qiskit.quantum_info import  Operator
from qiskit.circuit.library import  MCXGate, PhaseOracle
from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseOracleGate
from Chapter08_QuantumGates_functions import simulate_statevector,  simulate_measurements
import math

def bitstring_to_expression(bitstring: str):
    """Convert bitstring like '100' to PhaseOracleGate expression.
    Ensures variables appear in order x0, x1, ... so parse-order
    matches index order."""
    n = len(bitstring)
    terms = []
    for i in range(n):
        bit = bitstring[n - 1 - i]  # x0 is rightmost bit
        terms.append(f"x{i}" if bit == '1' else f"~x{i}")
    return " & ".join(terms)

def ensure_variable_order(expression, n):
    prefix = " & ".join(f"(x{i} | ~x{i})" for i in range(n))
    return prefix + " & " + expression

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
    for ii, bit in enumerate(reversed(s)):
        if bit == '1': 
            secretCircuit.cx(ii+1, 0)  # Control on qubit ii, target on ancilla 0
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

def createPhaseOracle(bitstring: str) -> PhaseOracleGate:
    """Create a PhaseOracleGate that marks |bitstring⟩.
    Reverses internally to account for Qiskit's little-endian ordering."""
    n = len(bitstring)
    terms = []
    for i, bit in enumerate(bitstring[::-1]):  # reverse the string
        var = f"x{i}"
        terms.append(var if bit == '1' else f"~{var}")
    expression = " & ".join(terms)
    return PhaseOracleGate(expression)

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


if __name__ == "__main__":
    U, n = bv_secret_circuit()
    circuit = QuantumCircuit(n+1,n)
    circuit.x(0) 
    circuit.h(0) # This brings qubit 0 to |-> state
    circuit.h(range(1,n+1)) 
    circuit.unitary(U,range(n+1),'Secret')
    circuit.h(range(1,n+1))
    circuit.measure(range(1,n+1), range(0,n)) 
    display(circuit.draw('mpl')) 
    counts = simulate_measurements(circuit,shots = 1)
    print(counts)
    print("The total depth is ", circuit.depth())
    print("The total width is ", circuit.width())
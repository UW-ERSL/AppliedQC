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
from qiskit import QuantumCircuit
from IPython.display import display
from qiskit.quantum_info import  Operator
from qiskit.circuit.library import  MCXGate, PhaseOracle
from typing import List, Dict,Tuple 
from Chapter08_QuantumGates_functions import (simulateCircuit, analyzeCircuitForHardware,
                                              runCircuitOnIBMQuantum)
class TrussFeasibilityGrover:
    """
    Generalized Grover search for any truss structure.
    
    User provides:
    1. Node coordinates
    2. Element connectivity
    3. Which nodes must have connections (constraints)
    """
    
    def __init__(self, 
                 elements: List[Tuple[int, int]],
                 constraint_nodes: List[int]):
        """
        Initialize general truss for Grover search.
        
        Parameters
        ----------
        elements : List[Tuple[int, int]]
            List of element connections as (node_i, node_j) tuples
            Example: [(0,1), (1,2), (3,4), (4,5), (0,3), ...]
            
        constraint_nodes : List[int]
            Node indices that must have at least one active element
            Example: [0, 4] means nodes 0 and 4 must each be connected

        """
        self.elements = elements
        self.n_elements = len(elements)
        self.constraint_nodes = constraint_nodes
        
        # Build element-to-node connectivity mapping
        self._build_node_element_map()
        
        # Calculate qubit requirements
        n_constraint_checks = len(constraint_nodes)

        
        # Qubit allocation
        self.element_qubits = list(range(self.n_elements))
        self.constraint_checks = list(range(self.n_elements, 
                                           self.n_elements + n_constraint_checks))
        self.output_qubit = self.n_elements + n_constraint_checks 
        self.n_qubits = self.output_qubit + 1
        
        print(f"Truss Configuration:")
        print(f"  Elements: {self.n_elements}")
        print(f"  Constraint nodes: {constraint_nodes}")
        print(f"  Total qubits: {self.n_qubits}")
    
    def _build_node_element_map(self):
        """Build mapping from constrained nodes to elements that connect to them."""
        self.node_elements = {}
        
        for node_id in self.constraint_nodes:
            connected_elements = []
            for elem_idx, (n1, n2) in enumerate(self.elements):
                if n1 == node_id or n2 == node_id:
                    connected_elements.append(elem_idx)
            self.node_elements[node_id] = connected_elements
    
    def _create_node_connectivity_circuit(self, element_list: List[int], 
                                         output_qubit: int) -> QuantumCircuit:
        """Create circuit to check if at least one element from list is active."""
        qc = QuantumCircuit(self.n_qubits)
        
        relevant_qubits = [self.element_qubits[elem] for elem in element_list]
        
        # Flip output if ALL elements are 0 (not connected)
        for q in relevant_qubits:
            qc.x(q)
        
        if len(relevant_qubits) == 1:
            qc.cx(relevant_qubits[0], output_qubit)
        else:
            qc.mcx(relevant_qubits, output_qubit)
        
        for q in relevant_qubits:
            qc.x(q)
        
        qc.x(output_qubit)
        
        return qc
    
    
    def create_oracle(self) -> QuantumCircuit:
        """Create oracle checking all node connectivity and symmetry constraints."""
        qc = QuantumCircuit(self.n_qubits)
        
        # 1. Create all node connectivity checks
        node_circuits = []
        for i, node_id in enumerate(self.constraint_nodes):
            node_circuit = self._create_node_connectivity_circuit(
                self.node_elements[node_id],
                self.constraint_checks[i]
            )
            node_circuits.append(node_circuit)
            qc.compose(node_circuit, inplace=True)
        
        
        # 3. Final Multi-Controlled X (The "AND" Gate)
        # All constraint checks 
        all_checks = self.constraint_checks 
        qc.mcx(all_checks, self.output_qubit)
        
        # 4. Uncompute to prevent entanglement issues
        for node_circuit in reversed(node_circuits):
            qc.compose(node_circuit.inverse(), inplace=True)
        
        return qc
    
    def create_grover_circuit(self, n_iterations: int = None,
                             decompose_mcx: bool = True) -> QuantumCircuit:
        """Create complete Grover search circuit."""
        oracle = self.create_oracle()
        
        if n_iterations is None:
            N = 2**self.n_elements
            M_estimate = 100  # Conservative estimate
            theta = math.asin(math.sqrt(M_estimate / N))
            n_iterations = int(math.pi / (4 * theta) - 0.5)
        
        qc = QuantumCircuit(self.n_qubits, self.n_elements)
        
        # Initialize output qubit for phase kickback
        qc.x(self.output_qubit)
        qc.h(self.output_qubit)
        
        # Create equal superposition
        qc.h(self.element_qubits)
        
        # Apply Grover iterations
        for _ in range(n_iterations):
            # Oracle
            qc.compose(oracle, inplace=True)
            
            # Diffusion
            qc.h(self.element_qubits)
            qc.x(self.element_qubits)
            qc.h(self.element_qubits[-1])
            qc.mcx(self.element_qubits[:-1], self.element_qubits[-1])
            qc.h(self.element_qubits[-1])
            qc.x(self.element_qubits)
            qc.h(self.element_qubits)
        
        qc.measure(self.element_qubits, range(self.n_elements))
        
        if decompose_mcx:
            qc = qc.decompose()
            qc = qc.decompose()
        
        return qc
    
    def verify_design(self, bitstring: str) -> Dict[str, bool]:
        """
        Verify if a design satisfies all constraints.
        Extracts the rightmost n_elements bits.
        """
        # Extract element bits (rightmost n_elements bits)
        element_bits = bitstring[-self.n_elements:]
        
        # Reverse so index matches qubit number
        design = [int(b) for b in element_bits[::-1]]
        
        # Check all node connectivity constraints
        node_checks = {}
        for node_id in self.constraint_nodes:
            node_ok = any(design[i] == 1 for i in self.node_elements[node_id])
            node_checks[f'node_{node_id}'] = node_ok
        
        
        # All constraints satisfied
        all_satisfied = all(node_checks.values()) 

        result = node_checks.copy()
        result['all_satisfied'] = all_satisfied
        
        return result
    

    def get_feasible_strings(self, counts: Dict[str, int], top_n: int = 1) -> List[Tuple[str, int]]:
        """Return top N designs from counts."""
        sorted_results = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return sorted_results
    
    def print_analysis(self, counts: Dict[str, int], top_n: int = 10):
        """Print analysis of results."""

        print(f"\n{'='*70}")
        print(f"Grover Search Results")
        print(f"{'='*70}")
        print(f"Total measurements: {sum(counts.values())}")
        print(f"Unique designs: {len(counts)}")
        
        # Calculate statistics
        verified_count = sum(count for bits, count in counts.items() 
                            if self.verify_design(bits)['all_satisfied'])
        
        print(f"Feasible designs: {verified_count}/{sum(counts.values())} "
              f"({100*verified_count/sum(counts.values()):.1f}%)")
        
        print(f"{'-'*70}")



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



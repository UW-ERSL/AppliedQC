"""
Chapter 11: Quantum Linear Algebra (QLA) Functions
"""

import numpy as np
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import  StatePreparation,DiagonalGate
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.library.standard_gates import PhaseGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def LCU_Ax(A, x, mode='statevector'):
    """Implements the LCU method to compute A|x> via Prep-Select-Unprep framework.
    
    Args:
        A (Operator): Hermitian operator
        x (np.ndarray): Input state vector (normalized)
        mode (str): 'statevector' or 'measurement'
        
    Returns:
        qc (QuantumCircuit): Quantum circuit implementing the LCU
        metadata (dict): Contains alpha, num_system, num_ancilla, initial_state
    """

    
    # Pauli decomposition
    pauli_split = SparsePauliOp.from_operator(A)
    coeffs = pauli_split.coeffs
    alpha = np.sum(np.abs(coeffs))

    L = len(coeffs)
    num_ancilla = int(np.ceil(np.log2(L)))
    num_system = int(np.ceil(np.log2(A.shape[0])))

    qr_sys = QuantumRegister(num_system, 's')
    qr_anc = QuantumRegister(num_ancilla, 'a')
    
    if mode == 'measurement':
        cr_anc = ClassicalRegister(num_ancilla, 'c_anc')
        cr_sys = ClassicalRegister(num_system, 'c_sys')
        qc = QuantumCircuit(qr_sys, qr_anc, cr_anc, cr_sys)
    else:
        qc = QuantumCircuit(qr_sys, qr_anc)

    # Initialize the circuit with |x>_sys ⊗ |0>_anc
    #qc.initialize(x, qr_sys)
    qc.append(StatePreparation(x), qr_sys)
    
    # PREP (on ancilla) - uses only magnitudes
    prep_vec = np.pad(np.sqrt(np.abs(coeffs) / alpha), (0, 2**num_ancilla - L))
    qc.append(StatePreparation(prep_vec, label='Prep'), qr_anc)

    # SELECT (controlled Paulis with phase incorporated)
    for i, (pauli, coeff) in enumerate(zip(pauli_split.paulis, coeffs)):
        # Extract phase from coefficient
        phase = np.angle(coeff)
        
        # Create a circuit for the Pauli with global phase
        pauli_circ = QuantumCircuit(num_system, global_phase=phase)
        pauli_circ.append(pauli.to_instruction(), range(num_system))
        
        # Convert to gate and add control
        pauli_gate_with_phase = pauli_circ.to_gate(label=f'{pauli}(φ={phase:.2f})')
        ctrl_gate = pauli_gate_with_phase.control(
            num_ancilla,
            ctrl_state=format(i, f'0{num_ancilla}b')
        )
        
        # Ancilla controls system: [anc, sys]
        qc.append(ctrl_gate, [*qr_anc, *qr_sys])

    # UNPREP (inverse on ancilla)
    qc.append(StatePreparation(prep_vec, label='Prep').inverse(), qr_anc)

    # Add measurements if in measurement mode
    if mode == 'measurement':
        qc = qc.decompose(reps=3)  # Decompose multiple times to get to basis gates
        qc.measure(qr_anc, cr_anc)
        qc.measure(qr_sys, cr_sys)


    # Return circuit and metadata
    metadata = {
        'alpha': alpha,
        'num_system': num_system,
        'num_ancilla': num_ancilla,
        'coeffs': coeffs,
        'pauli_split': pauli_split
    }
    
    return qc, metadata

def Pauli_Block_Encoding(A, mode='statevector'):
    """Constructs a block-encoding of a Hermitian operator A using Pauli decomposition.
    
    Args:
        A (Operator): Hermitian operator    
    Returns:
        qc (QuantumCircuit): Quantum circuit implementing the block-encoding        
    """
    x = np.zeros(A.shape[0])# dummy
    x[0] = 1.0  # |0> state
    qc, metadata = LCU_Ax(A, x, mode=mode)
    U_matrix = Operator(qc).data
    return U_matrix, metadata

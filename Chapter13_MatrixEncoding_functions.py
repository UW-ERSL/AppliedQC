"""
Chapter 11: Quantum Encoding Functions
"""

import numpy as np
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import StatePreparation, DiagonalGate
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.library.standard_gates import PhaseGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from Chapter08_QuantumGates_functions import (simulate_statevector, simulate_measurements, runCircuitOnIBMQuantum)


def LCU_Ax(A, x, mode='statevector'):
    """Implements the LCU method to compute A|x> via Prep-Select-Unprep framework.

    Convention: system qubits are least significant bits in statevector (declared
    first in QuantumCircuit), ancilla qubits are most significant bits (declared
    second). Post-selection on ancilla=|0> extracts the first 2**num_system entries
    of the statevector.

    Args:
        A (np.ndarray): Hermitian operator
        x (np.ndarray): Input state vector (normalized)
        mode (str): 'statevector' or 'measurement'

    Returns:
        qc (QuantumCircuit): Quantum circuit implementing the LCU
        metadata (dict): Contains alpha, num_system, num_ancilla, coeffs, pauli_split
    """

    # Pauli decomposition
    pauli_split = SparsePauliOp.from_operator(A)
    coeffs = pauli_split.coeffs
    alpha = np.sum(np.abs(coeffs))

    L = len(coeffs)
    num_ancilla = int(np.ceil(np.log2(L)))
    num_system = int(np.ceil(np.log2(A.shape[0])))

    # Ancilla declared first -> least significant bits in statevector
    # System declared second -> most significant bits in statevector
    qr_anc = QuantumRegister(num_ancilla, 'a')
    qr_sys = QuantumRegister(num_system, 's')
    
    if mode == 'measurement':
        cr_sys = ClassicalRegister(num_system, 'c_sys')
        cr_anc = ClassicalRegister(num_ancilla, 'c_anc')
        qc = QuantumCircuit(qr_anc, qr_sys, cr_anc, cr_sys)
    else:
        qc = QuantumCircuit(qr_anc,qr_sys) 

    # Initialize |x>_sys ⊗ |0>_anc
    qc.append(StatePreparation(x), qr_sys)

    # PREP on ancilla - encode sqrt(|c_k| / alpha)
    prep_vec = np.pad(np.sqrt(np.abs(coeffs) / alpha), (0, 2**num_ancilla - L))
    qc.append(StatePreparation(prep_vec, label='Prep'), qr_anc)

    # SELECT: ancilla controls system
    # ctrl_gate qubit order: [control_qubits..., target_qubits...]
    # => [*qr_anc, *qr_sys]: ancilla=control, system=target  ✓
    for i, (pauli, coeff) in enumerate(zip(pauli_split.paulis, coeffs)):
        phase = np.angle(coeff)
        pauli_circ = QuantumCircuit(num_system, global_phase=phase)
        pauli_circ.append(pauli.to_instruction(), range(num_system))
        ctrl_gate = pauli_circ.to_gate(
            label=f'{pauli}(φ={phase:.2f})'
        ).control(
            num_ancilla,
            ctrl_state=format(i, f'0{num_ancilla}b')
        )
        # control=ancilla, target=system
        qc.append(ctrl_gate, [*qr_anc, *qr_sys])

    # UNPREP: inverse PREP on ancilla
    qc.append(StatePreparation(prep_vec, label='Prep').inverse(), qr_anc)

    # ancilla is least significant => ancilla=|0> 
    if mode == 'measurement':
        qc = qc.decompose(reps=3)
        qc.measure(qr_sys, cr_sys)
        qc.measure(qr_anc, cr_anc)

    metadata = {
        'alpha': alpha,
        'num_system': num_system,
        'num_ancilla': num_ancilla,
        'coeffs': coeffs,
        'pauli_split': pauli_split,
        'ancilla_zero_stride': 2**num_ancilla,
    }
    return qc, metadata


def Pauli_Block_Encoding(A, mode='statevector'):
    """Constructs a block-encoding of a Hermitian operator A using Pauli decomposition.
    
    Args:
        A (np.ndarray): Hermitian operator
        mode (str): 'statevector' or 'measurement'

    Returns:
        U_matrix (np.ndarray): Unitary matrix of the block-encoding
        metadata (dict): Contains alpha, num_system, num_ancilla, coeffs, pauli_split
    """
    x = np.zeros(A.shape[0])  # dummy state vector
    x[0] = 1.0                # |0> state
    qc, metadata = LCU_Ax(A, x, mode=mode)
    U_matrix = Operator(qc).data
    return U_matrix, metadata


def LCU_fTAx(f, A, x, shots=10000, noise_model=None):
    """
    Compute f^T * A * x by extending the LCU_Ax circuit.

    Args:
        f (np.ndarray): Observable vector (will be normalized)
        A (np.ndarray): Matrix
        x (np.ndarray): Input vector
        shots (int): Number of measurements

    Returns:
        inner_product (float): Estimate of |f^T * A * x|
        qc (QuantumCircuit): Full measurement circuit
        metadata (dict): Circuit metadata including success_prob
    """

    # Normalize f
    f = f / np.linalg.norm(f)

    # Step 1: Get the base LCU circuit (statevector mode = no measurements)
    qc, metadata = LCU_Ax(A, x, mode='statevector')
   
    num_system = metadata['num_system']
    num_ancilla = metadata['num_ancilla']

    # Step 2: Get register references by name (robust to ordering)
    qr_anc = qc.qregs[0]  # ancilla: declared first in QuantumCircuit(qr_anc, qr_sys)
    qr_sys = qc.qregs[1]  # system:  declared second

    # Step 3: Add classical registers
    cr_anc = ClassicalRegister(num_ancilla, 'c_anc')
    cr_sys = ClassicalRegister(num_system, 'c_sys')
    qc.add_register(cr_anc)
    qc.add_register(cr_sys)

    # Step 4: Measure ancilla for post-selection
    qc.measure(qr_anc, cr_anc)
    qc.barrier() # This is important
    
    # Step 5: Add f-basis rotation on system
    Uf_gate = StatePreparation(f, label='f').inverse()
    
    qc.append(Uf_gate, qr_sys)

    # Step 6: Measure system
    qc.measure(qr_sys, cr_sys)
   
    # Step 7: Run circuit
    counts = simulate_measurements(qc, shots=shots, noise_model=noise_model)
   
    # Step 8: Post-process
    ancilla_zero = '0' * num_ancilla
    system_zero  = '0' * num_system
    alpha = metadata['alpha']

    count_proj = 0
    total_postselected = 0

    for outcome, count in counts.items():
        # Qiskit bit string order: 'c_sys c_anc' (last register is leftmost)
        parts = outcome.split(' ')
        sys_bits = parts[0]   # c_sys is added second -> leftmost in string
        anc_bits = parts[1]   # c_anc is added first  -> rightmost in string

        if anc_bits == ancilla_zero:
            total_postselected += count
            if sys_bits == system_zero:
                count_proj += count

    success_prob = total_postselected / shots
    metadata['success_prob'] = success_prob

    if total_postselected > 0:
        prob_f = count_proj / total_postselected
        norm_Ax = alpha * np.sqrt(success_prob)
        inner_product = np.sqrt(prob_f) * norm_Ax
        return inner_product, qc, metadata
    else:
        return 0.0, qc, metadata
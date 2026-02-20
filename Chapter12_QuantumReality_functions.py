
import numpy as np  
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from Chapter11_QuantumEncoding_functions import (LCU_Ax)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import StatePreparation


def postprocess_measurements(counts, metadata):
    """Post-process measurement results.
    
    NOTE: Measurements only give MAGNITUDES, not signs or phases!
    
    Args:
        counts (dict): Measurement counts from circuit execution
        metadata (dict): Metadata from LCU_Ax
        
    Returns:
        res_vector (np.ndarray): Resulting state vector (magnitudes only)
        success_prob (float): Post-selection success probability
    """
    num_system = metadata['num_system']
    num_ancilla = metadata['num_ancilla']
    alpha = metadata['alpha']
    
    # Post-select: keep only measurements where ancilla = 0...0
    ancilla_zero_pattern = '0' * num_ancilla
    res_counts = np.zeros(2**num_system, dtype=float)
    total_shots = sum(counts.values())
    total_postselected = 0
    
    for outcome, count in counts.items():
        parts = outcome.split(' ')
        sys_bits = parts[0]  # c_sys
        anc_bits = parts[1]  # c_anc
        
        # Check if ancilla is all zeros
        if anc_bits == ancilla_zero_pattern:
            sys_idx = int(sys_bits, 2)
            res_counts[sys_idx] += count
            total_postselected += count
    
    # Success probability
    success_prob = total_postselected / total_shots
    
    if total_postselected > 0:
        # Convert counts to probability distribution
        prob_dist = res_counts / total_postselected
        
        # Take sqrt to get amplitudes (magnitudes only)
        res_vector = np.sqrt(prob_dist)
        
        # Scale by alpha and sqrt(success_prob)
        # The sqrt(success_prob) accounts for the post-selection normalization
        res_vector *= alpha * np.sqrt(success_prob)
    else:
        res_vector = np.zeros(2**num_system, dtype=float)
    
    return res_vector, success_prob

def LCU_fTAx(f, A, x, shots=10000):
    """
    Compute f^T * A * x by extending the LCU_Ax circuit
    
    Strategy: Create LCU circuit, then add f-basis rotation and measurements
    
    Args:
        f: Observable vector
        A: Matrix
        x: Input vector
        shots: Number of measurements
        
    Returns:
        inner_product: |f^T * A * x| (magnitude)
        success_prob: Post-selection probability
    """

    # Normalize f
    f = f / np.linalg.norm(f)
    
    # Step 1: Get the base LCU circuit (without measurements)
    qc, metadata = LCU_Ax(A, x, mode='statevector')  # Use statevector mode to avoid measurements
    
    # Step 2: Add classical registers for measurements
    num_system = metadata['num_system']
    num_ancilla = metadata['num_ancilla']
    
    cr_anc = ClassicalRegister(num_ancilla, 'c_anc')
    cr_sys = ClassicalRegister(num_system, 'c_sys')
    qc.add_register(cr_anc, cr_sys)
    
    # Get register references
    qr_sys = qc.qregs[0]  # System register (first in LCU_Ax)
    qr_anc = qc.qregs[1]  # Ancilla register (second in LCU_Ax)
    
    # Step 3: Measure ancilla for post-selection
    qc.measure(qr_anc, cr_anc)
    qc.barrier()
    
    # Step 4: Add f-basis rotation
    f_basis_gate = StatePreparation(f, label='f').inverse()
    qc.append(f_basis_gate, qr_sys)
    
    # Step 5: Measure system
    qc.measure(qr_sys, cr_sys)
    qc.draw('mpl')
    plt.show()
   
    # Step 6: Run and post-process
    simulator = AerSimulator()
    qc_trans = transpile(qc.decompose(reps=5), simulator, optimization_level=1)
    result = simulator.run(qc_trans, shots=shots).result()
    counts = result.get_counts()
    
    # Post-process (same as before)
    ancilla_zero = '0' * num_ancilla
    system_zero = '0' * num_system
    alpha = metadata['alpha']
    
    count_proj = 0
    total_postselected = 0
    
    for outcome, count in counts.items():
        parts = outcome.split(' ')
        sys_bits = parts[0]
        anc_bits = parts[1]
        
        if anc_bits == ancilla_zero:
            total_postselected += count
            if sys_bits == system_zero:
                count_proj += count
    
    success_prob = total_postselected / shots
    
    if total_postselected > 0:
        prob_f = count_proj / total_postselected
        norm_Ax = alpha * np.sqrt(success_prob)
        inner_product = np.sqrt(prob_f) * norm_Ax
        return inner_product, success_prob
    else:
        return 0, 0
    

def execute_fTAx_from_result(result, metadata):
    """
    Extract f^T * A * x from an existing simulation result
    
    Args:
        result: Qiskit result object
        metadata: Circuit metadata
        
    Returns:
        inner_product: |f^T * A * x|
        success_prob: Post-selection probability
    """
    counts = result.get_counts()
    
    num_system = metadata['num_system']
    num_ancilla = metadata['num_ancilla']
    alpha = metadata['alpha']
    
    ancilla_zero = '0' * num_ancilla
    system_zero = '0' * num_system
    
    count_proj = 0
    total_postselected = 0
    total_shots = sum(counts.values())
    
    for outcome, count in counts.items():
        parts = outcome.split(' ')
        sys_bits = parts[0]
        anc_bits = parts[1]
        
        if anc_bits == ancilla_zero:
            total_postselected += count
            if sys_bits == system_zero:
                count_proj += count
    
    success_prob = total_postselected / total_shots
    
    if total_postselected > 0:
        prob_f = count_proj / total_postselected
        norm_Ax = alpha * np.sqrt(success_prob)
        inner_product = np.sqrt(prob_f) * norm_Ax
        return inner_product, success_prob
    else:
        return 0, 0
# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Define problem
    A = np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0.5, 0, 0, 1]
    ], dtype=float)
    
    x = np.array([0.6, 0.8, 0, 0], dtype=complex)
    f = np.array([1, 0, 0.5, 0.2])
    
    # Classical computation
    expected = np.dot(f, A @ x)
    
    print("="*60)
    print("Computing f^T * A * x")
    print("="*60)
    print(f"Expected: {expected:.4f}")
    print(f"Expected magnitude: {np.abs(expected):.4f}")
    
    # Quantum computation
    result, prob = LCU_fTAx(f, A, x, shots=10000)
    
    print(f"\nQuantum result: {result:.4f}")
    print(f"Error: {abs(result - np.abs(expected)):.4f}")
    print(f"Success probability: {prob:.4f}")
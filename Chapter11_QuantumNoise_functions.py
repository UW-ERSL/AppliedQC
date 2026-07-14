"""
Quantum Noise and LCU Measurement Post-Processing
=================================================
Post-processing helpers for circuits that implement matrix-vector products
via a Linear Combination of Unitaries (LCU) with an ancilla-based block
encoding, as used in this chapter's quantum-noise experiments.

Because measurement yields only outcome probabilities, these routines recover
state-vector magnitudes (never signs or phases): they post-select on the
all-zero ancilla pattern, renormalize by the block-encoding factor alpha and
the post-selection success probability, and report the resulting amplitudes
and post-selection success rate.
"""

import numpy as np  
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
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

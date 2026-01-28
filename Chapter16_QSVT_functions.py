import numpy as np
import scipy
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator

from numpy.polynomial.chebyshev import Chebyshev
from pyqsp.angle_sequence import QuantumSignalProcessingPhases 

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev

def convert_qiskit_to_reflection(qiskit_phases):
    """
    Converts Qiskit 'Wz' convention phases to the Reflection convention 
    used in the Sünderhauf / Gilyén papers.
    """
    phases = np.array(qiskit_phases)
    d = len(phases) - 1  # Degree of polynomial
    reflection_phases = np.zeros_like(phases)
    
    # 1. First angle adjustment
    reflection_phases[0] = phases[0] + np.pi/4
    
    # 2. Middle angles adjustment
    for j in range(1, d):
        reflection_phases[j] = phases[j] + np.pi/2
        
    # 3. Last angle adjustment
    reflection_phases[d] = phases[d] + np.pi/4
    
    # Normalize phases to be within [-pi, pi]
    return (reflection_phases + np.pi) % (2 * np.pi) - np.pi

def plot_approximation_comparison(degree, kappa):
    """Visualization showing how δ/x blows up near the singularity."""
    phases, scale_factor, sigma = get_inverse_phases(degree, kappa, buffer=0)
    
    delta = 1 / kappa
    x = np.linspace(-1, 1, 2000)
    
    # Define how close to show the singularity
    x_near_singularity = 0.1 / kappa
    
    def target_scaled(x):
        x_safe = np.where(np.abs(x) < 1e-15, 1e-15 * np.sign(x), x)
        return scale_factor / x_safe
    
    def target_smooth(x):
        x_safe = np.where(np.abs(x) < 1e-15, 1e-15 * np.sign(x), x)
        return scale_factor * (1 - np.exp(-(x_safe / sigma)**2)) / x_safe
    
    poly = Chebyshev.interpolate(target_smooth, deg=degree, domain=[-1, 1])
    poly.coef[::2] = 0
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Show scaled ideal from ±0.1/κ onwards
    mask_pos = x > x_near_singularity
    mask_neg = x < -x_near_singularity
    
    ax.plot(x[mask_pos], target_scaled(x)[mask_pos], 'k--', linewidth=2.5, alpha=0.6,
            label=rf'Scaled ideal')
    ax.plot(x[mask_neg], target_scaled(x)[mask_neg], 'k--', linewidth=2.5, alpha=0.6)
    
    # Plot smoothed and polynomial
    ax.plot(x, target_smooth(x), 'b-', linewidth=2.5, 
            label=f'Smoothed')
    ax.plot(x, poly(x), 'r--', linewidth=2, alpha=0.8,
            label=f'Polynomial')
    
    # Shading
    ax.axvspan(delta, 1, alpha=0.15, color='green', 
               label=f'Valid region')
    ax.axvspan(-1, -delta, alpha=0.15, color='green')
    
    # Mark boundaries
    ax.axvline(delta, color='orange', linestyle='--', linewidth=1.5, 
               label=f'δ = {delta:.3f}')
    ax.axvline(-delta, color='orange', linestyle='--', linewidth=1.5)
    
    ax.axvline(x_near_singularity, color='gray', linestyle=':', 
               linewidth=1, alpha=0.5)
    ax.axvline(-x_near_singularity, color='gray', linestyle=':', 
               linewidth=1, alpha=0.5)
    
    ax.set_xlabel('x (normalized singular value)', fontsize=20)
    ax.set_ylabel('f(x)', fontsize=20)
    ax.set_title(f'Scaled Inverse Approximation (κ={kappa}, d={degree})', 
                 fontsize=20, fontweight='bold')
    ax.legend(fontsize=20, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Auto y-limits based on value at cutoff
    max_at_cutoff = abs(scale_factor / x_near_singularity)
    ylim = min(max_at_cutoff * 1.2, 10)  # Cap at 10 for readability
    ax.set_ylim([-ylim, ylim])
    

    
    plt.tight_layout()
    plt.show()

# Usag

def _reconstruct_polynomial_from_phases(degree, scale_factor, sigma):
    """
    Helper: Reconstruct Chebyshev polynomial from smoothing parameters.
    
    This ensures consistency across all error computation functions.
    """
    def target_smooth(x):
        x_safe = np.where(np.abs(x) < 1e-15, 1e-15 * np.sign(x), x)
        return scale_factor * (1 - np.exp(-(x_safe / sigma)**2)) / x_safe
    
    poly = Chebyshev.interpolate(target_smooth, deg=degree, domain=[-1, 1])
    poly.coef[::2] = 0  # Enforce odd parity
    
    return poly


def get_total_error_vs_unscaled_ideal(degree, kappa, buffer=0):
    """
    Computes total approximation error: |(1/δ)P(x) - 1/x|
    """
    phases, scale_factor, sigma = get_inverse_phases(degree, kappa, buffer)
    
    kappa_eff = kappa * (1 + buffer)
    delta_eff = 1 / kappa_eff
    
    x_pos = np.linspace(delta_eff, 1, 10000)
    x_neg = np.linspace(-1, -delta_eff, 10000)
    
    # Use helper to reconstruct polynomial
    poly = _reconstruct_polynomial_from_phases(degree, scale_factor, sigma)
    
    # Scale to approximate 1/x (not δ/x)
    y_poly_scaled_pos = poly(x_pos) / scale_factor
    y_poly_scaled_neg = poly(x_neg) / scale_factor
    
    y_ideal_pos = 1 / x_pos
    y_ideal_neg = 1 / x_neg
    
    error_pos = np.abs(y_poly_scaled_pos - y_ideal_pos)
    error_neg = np.abs(y_poly_scaled_neg - y_ideal_neg)
    
    return max(np.max(error_pos), np.max(error_neg))


def get_inverse_phases(degree, kappa=2, buffer=0):
    """
    Generates QSVT phase angles for f(x) ≈ 1/x approximation.
    
    Uses smoothed inverse with Gaussian fade to handle singularity,
    following QSVT framework (Gilyen et al., 2019, STOC).
    
    Args:
        degree (int): Polynomial degree (will be made odd for parity)
        kappa (float): Condition number κ = σ_max/σ_min
        buffer (float): Safety margin to push singularity outside domain
    
    Returns:
        phases (list): QSVT phase angles
        scale_factor (float): Normalization constant
        sigma (float): Smoothing parameter (for error computation)
    """
    if degree % 2 == 0:
        degree += 1
        
    kappa_eff = kappa * (1 + buffer)
    delta_eff = 1 / kappa_eff
    scale_factor = delta_eff
    
    # Define sigma ONCE - this is your smoothing parameter
    sigma = delta_eff / 3  # Consistent with 1/(3*kappa_eff)
    
    def target_func(x):
        x_safe = np.where(np.abs(x) < 1e-15, 1e-15 * np.sign(x), x)
        return scale_factor * (1 - np.exp(-(x_safe / sigma)**2)) / x_safe

    poly = Chebyshev.interpolate(target_func, deg=degree, domain=[-1, 1])
    poly.coef[::2] = 0  # Odd parity
    
    max_val = np.max(np.abs(poly(np.linspace(-1, 1, 1000))))
    if max_val > 0.999:
        poly.coef *= (0.999 / max_val)
        scale_factor *= (0.999 / max_val)

    phases = QuantumSignalProcessingPhases(poly, signal_operator="Wz")
    
    return [float(phi) for phi in phases], scale_factor, sigma


def get_error_inverse_approximation(degree, kappa=2, buffer=0):
    """
    Computes approximation error for QSVT inverse polynomial.
    
    Returns:
        max_error (float): Maximum absolute error in [δ,1] ∪ [-1,-δ]
    """
    phases, scale_factor, sigma = get_inverse_phases(degree, kappa, buffer)
    
    kappa_eff = kappa * (1 + buffer)
    delta_eff = 1 / kappa_eff
    
    # Sample in valid regions only
    x_pos = np.linspace(delta_eff, 1, 10000)
    x_neg = np.linspace(-1, -delta_eff, 10000)
    
    def target_func(x):
        x_safe = np.where(np.abs(x) < 1e-15, 1e-15 * np.sign(x), x)
        return scale_factor * (1 - np.exp(-(x_safe / sigma)**2)) / x_safe
    
    # Reconstruct polynomial from phases (more efficient than re-interpolating)
    poly = Chebyshev.interpolate(target_func, deg=degree, domain=[-1, 1])
    poly.coef[::2] = 0
    
    error_pos = np.max(np.abs(poly(x_pos) - target_func(x_pos)))
    error_neg = np.max(np.abs(poly(x_neg) - target_func(x_neg)))
    
    return max(error_pos, error_neg)


class myQSVT:
    """
    QSVT Algorithm for Linear Systems (2x2 Matrix)
    =============================================
    Solves Ax = b by approximating f(A) = A⁻¹ using polynomial transformation
    of singular values.
    """
    def __init__(self, A, b, degree=15, kappa=4, nShots=1000):
        self.A = A
        self.b = b
        self.nShots = nShots
        self.n = int(np.log2(len(b))) # Vector qubits (1 for 2x2)
        self.dataOK = self._validate_input()
        
        # Block Encoding requires 1 ancilla to embed 2x2 in 4x4
        self.ancilla_qubits = 1 

        degree = degree  # You can now set this to any odd integer
        kappa = kappa    
        self.angles, self.tau, self.sigma  = get_inverse_phases(degree,kappa)
        print(f"Generated {len(self.angles)} angles for degree {degree}")
    def _validate_input(self):
        # Ensure singular values < 1
        s = np.linalg.svd(self.A, compute_uv=False)
        if np.any(s >= 1):
            print(f"Warning: Singular values {s} must be < 1 for block encoding.")
            return False
        return True

    def get_block_encoding(self):
        """
        Embeds A into the top-left block of a 4x4 unitary U.
        U = [[A, sqrt(I - AA†)], [sqrt(I - A†A), -A†]]
        """
        N = self.A.shape[0]
        I = np.eye(N)
        A_dag = self.A.conj().T
        
        # Construct the components of the unitary dilation
        top_right = scipy.linalg.sqrtm(I - self.A @ A_dag)
        bottom_left = scipy.linalg.sqrtm(I - A_dag @ self.A)
        bottom_right = -A_dag
        
        # Build the 4x4 matrix
        U_matrix = np.block([
            [self.A, top_right],
            [bottom_left, bottom_right]
        ])
        return Operator(U_matrix)

    def apply_projector_phase(self, circuit, phi, qubits):
        """Applies the Rz-like phase shift (e^{i * phi * (2|0><0| - I)})"""
        # This is equivalent to a multi-controlled-Z with phases
        circuit.x(qubits[0])
        circuit.rz(2 * phi, qubits[0])
        circuit.x(qubits[0])

    def construct_qsvt_circuit(self):
        """
        Builds the QSVT sequence with corrected qubit mapping and loop.
        """
        q_data = QuantumRegister(self.n, 'b')
        q_anc = QuantumRegister(self.ancilla_qubits, 'anc')
        c = ClassicalRegister(self.n + self.ancilla_qubits, 'meas')
        qc = QuantumCircuit(q_anc, q_data, c)
        
        qc.prepare_state(Statevector(self.b), q_data)
        qc.barrier()

        U_op = self.get_block_encoding()
        U_gate = U_op.to_instruction()
        U_dag_gate = U_op.adjoint().to_instruction()

        # CORRECTED LOOP: Apply d unitaries and d+1 phases
        # For degree 15, angles has 16 elements. 
        # We loop 15 times for (Phase -> Unitary) and then apply one final Phase.
        for i in range(len(self.angles) - 1):
            self.apply_projector_phase(qc, self.angles[i], q_anc)
            
            # SWAP QUBITS HERE: [q_data, q_anc] makes q_anc the MSB (index 2,3)
            # This aligns the "top-left" block of U with the ancilla=0 subspace.
            if i % 2 == 0:
                qc.append(U_gate, [q_data[0], q_anc[0]]) 
            else:
                qc.append(U_dag_gate, [q_data[0], q_anc[0]])
        
        # Final Phase
        self.apply_projector_phase(qc, self.angles[-1], q_anc)
        
        qc.barrier()
        qc.measure(range(qc.num_qubits), range(qc.num_qubits))

        print(f"Circuit width (total qubits): {qc.width()}")
        print(f"Circuit depth: {qc.depth()}")
        return qc

    def execute(self):
        if not self.dataOK: return None
        
        qc = self.construct_qsvt_circuit()
        backend = Aer.get_backend('qasm_simulator')
        t_qc = transpile(qc, backend)
        counts = backend.run(t_qc, shots=self.nShots).result().get_counts()
        
        # Post-selection: Since q_anc is index 0, success is bitstrings ending in '0'
        success_counts = {k: v for k, v in counts.items() if k.endswith('0')}
        
        total_success = sum(success_counts.values())
        if total_success == 0:
            return np.zeros(2**self.n)
            
        u_qsvt = np.zeros(2**self.n)
        for bitstr, count in success_counts.items():
            # bitstr is "q_data q_anc" -> "1 0" or "0 0"
            # The data bit is the first character
            idx = int(bitstr[0])
            u_qsvt[idx] = np.sqrt(count / total_success)
            
        return u_qsvt / np.linalg.norm(u_qsvt)
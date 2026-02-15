import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator
from numpy.polynomial.chebyshev import Chebyshev
from pyqsp.angle_sequence import QuantumSignalProcessingPhases 
from numpy.polynomial import Chebyshev

def qsp_unitary_reflection(x, phases):
    """
    Computes the QSP unitary using the Reflection convention.
    W = R(phi_0) Z(x) R(phi_1) Z(x) ... R(phi_d)
    where R(phi) = exp(i * phi * Z) and Z(x) is the signal operator.
    """
    # Signal operator for a scalar x
    def U_x(x_val):
        return np.array([[x_val, np.sqrt(1-x_val**2)], 
                         [np.sqrt(1-x_val**2), -x_val]])

    # Reflection operator e^{i * phi * Z}
    def R_phi(phi):
        return np.array([[np.exp(1j * phi), 0], 
                         [0, np.exp(-1j * phi)]])

    # Build the circuit: W = R(phi_0) * U(x) * R(phi_1) * ...
    # Note: Conventions vary on whether you start with R or U. 
    # This follows the Gilyen/QSVT standard.
    W = R_phi(phases[0])
    for phi in phases[1:]:
        W = W @ U_x(x) @ R_phi(phi)
    
    return W[0, 0] # The polynomial P(x) is the top-left entry

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

class SunderhaufPolynomial:
    """Class for computing optimal polynomial approximating 1/x"""
    
    @staticmethod
    def helper_Lfrac(n: int, x: float, a: float) -> float:
        """Compute mathcal{L}_n(x; a)"""
        alpha = (1+a)/(2*(1-a))
        l1 = (x + (1-a)/(1+a)) / alpha
        l2 = (x**2 + (1-a)/(1+a) * x / 2 - 1/2) / alpha**2
        if n == 1:
            return l1
        for _ in range(3, n+1):
            l1, l2 = l2, x * l2 / alpha - l1 / (4 * alpha**2)
        return l2

    @staticmethod
    def helper_P(x: float, n: int, a: float) -> float:
        """Compute values of the polynomial P_(2n-1)(x; a)"""
        return (1 - (-1)**n * (1+a)**2 / (4*a) * SunderhaufPolynomial.helper_Lfrac(n, (2 * x**2 - (1 + a**2))/(1-a**2), a))/x

    @staticmethod
    def poly(d: int, a: float) -> np.polynomial.Chebyshev:
        """Returns Chebyshev polynomial for optimal polynomial
        
        Args:
            d (int): odd degree
            a (float): 1/kappa for range [a,1]"""
        if d % 2 == 0:
            raise ValueError("d must be odd")
        coef = np.polynomial.chebyshev.chebinterpolate(
            SunderhaufPolynomial.helper_P, d, args=((d+1)//2, a))
        coef[0::2] = 0  # force even coefficients exactly zero
        return np.polynomial.Chebyshev(coef)

    @staticmethod
    def error_for_degree(d: int, a: float) -> float:
        """Returns the poly error for degree d and a=1/kappa"""
        if d % 2 == 0:
            raise ValueError("d must be odd")
        n = (d+1)//2
        return (1-a)**n / (a * (1+a)**(n-1))

    @staticmethod
    def mindegree(epsilon: float, a: float) -> int:
        """Returns the minimum degree d for a poly with error epsilon, a=1/kappa"""
        n = math.ceil((np.log(1/epsilon) + np.log(1/a) + np.log(1+a))
                      / np.log((1+a) / (1-a)))
        return 2*n-1


class myQSVT:
    def __init__(self, A, b, degree=15, kappa=None, nShots=1000, method='sunderhauf', target_error=None):
        """
        Args:
            kappa: Condition number (if None, auto-compute from A)
            target_error: Target approximation error (for Sunderhauf method)
                        If specified, ignores 'degree' parameter
        """
        self.A = A
        self.b = b
        self.nShots = nShots
        self.n = int(np.log2(len(b)))
        self.ancilla_qubits = 1
        self.method = method
        
        # Compute actual condition number
        s = np.linalg.svd(A, compute_uv=False)
        print(f"Singular values: {s}")
        self.actual_kappa = s[0] / s[-1]
        
        # Use provided kappa or actual
        if kappa is None:
            self.kappa = self.actual_kappa
            print(f"Auto-detected κ = {self.kappa:.2f}")
        else:
            self.kappa = kappa
            if abs(self.kappa - self.actual_kappa) > 0.1 * self.actual_kappa:
                print(f"Warning: Specified κ={kappa:.2f} differs from actual κ={self.actual_kappa:.2f}")
        
        self.dataOK = self._validate_input()
  
        self.degree = degree
        self.target_error = target_error

        if method == 'sunderhauf':
            self.angles, self.tau, self.achieved_error = self.get_inverse_phases_sunderhauf(
                self.kappa, target_error=target_error)
        elif method == 'interpolation':
            self.angles, self.tau, self.sigma = self.get_inverse_phases_interpolation(degree, self.kappa)
        else:
            raise ValueError("method must be 'sunderhauf' or 'interpolation'")
            
        print(f"Method: {method}")
        print(f"Generated {len(self.angles)} angles for degree {len(self.angles)-1}")

    def get_inverse_phases_sunderhauf(self, kappa, target_error=None):
        """
        Get QSVT phases using Sunderhauf's optimal polynomial
        
        Args:
            kappa: Condition number
            target_error: Desired approximation error (optional)
                         If None, uses degree from initialization
        
        Returns:
            phases: QSVT phase angles
            tau: Scaling factor (equivalent to delta_eff)
            achieved_error: Actual approximation error
        """
        a = 1 / kappa
        
        # Option 1: User specifies target error - compute optimal degree
        if target_error is not None:
            degree = SunderhaufPolynomial.mindegree(target_error, a)
            print(f"Optimal degree for ε={target_error:.2e}: {degree}")
            self.degree = degree
        # Option 2: User specifies degree - compute achieved error
        else:
            degree = self.degree
            if degree % 2 == 0:
                degree += 1
            achieved_error = SunderhaufPolynomial.error_for_degree(degree, a)
            print(f"Using degree {degree}, achieved error: {achieved_error:.2e}")
        
        # Get optimal polynomial
        poly = SunderhaufPolynomial.poly(degree, a)
        
        # Compute achieved error
        achieved_error = SunderhaufPolynomial.error_for_degree(degree, a)
        
        # Normalize polynomial to satisfy ||p|| <= 1
        # Section III of Sunderhauf paper
        M = self._compute_polynomial_max(poly, degree)
        print(f"Polynomial maximum: {M:.4f}")
        
        # Scale factor: need to normalize by (kappa + error) and by M
        # tau is the effective scaling (like delta_eff in original code)
        tau = (kappa + achieved_error) / M
        
        # Scale polynomial coefficients
        poly_normalized = Chebyshev(poly.coef / M)
        
        # Verify normalization
        max_val = np.max(np.abs(poly_normalized(np.linspace(-1, 1, 1000))))
        print(f"After normalization, max value: {max_val:.6f}")
        
        if max_val > 0.999:
            print(f"Warning: polynomial maximum {max_val} is close to 1, adding safety margin")
            poly_normalized = Chebyshev(poly_normalized.coef * 0.999 / max_val)
            tau *= 0.999 / max_val
        
        # Generate QSVT phases
        phases = QuantumSignalProcessingPhases(poly_normalized, signal_operator="Wx")
        
        return [float(phi) for phi in phases], tau, achieved_error

    def _compute_polynomial_max(self, poly, degree):
        """
        Compute maximum value of polynomial using Sunderhauf's method (Section III)
        
        Uses Eq. (25-26) from the paper for rigorous bound
        """
        N = 25 * degree  # Sunderhauf's choice
        x_samples = np.linspace(-1, 1, N)
        p_values = np.abs(poly(x_samples))
        max_sampled = np.max(p_values)
        
        # Correction factor for rigorous bound (Eq. 25)
        correction = 1 / np.cos(np.pi * degree / (2 * N))
        M = max_sampled * correction
        
        return M

    def get_inverse_phases_interpolation(self, degree, kappa=2, buffer=0):
        """
        Original method: Get QSVT phases using smoothed interpolation
        
        This uses a Gaussian-smoothed 1/x function to avoid singularity
        """
        if degree % 2 == 0:
            degree += 1
            
        delta_eff = 1 / (kappa * (1 + buffer))
        sigma = delta_eff / 5
        
        def target_func(x):
            x_safe = np.where(np.abs(x) < 1e-15, 1e-15 * np.sign(x), x)
            return delta_eff * (1 - np.exp(-(x_safe / sigma)**2)) / x_safe

        poly = Chebyshev.interpolate(target_func, deg=degree, domain=[-1, 1])
        poly.coef[::2] = 0
        
        max_val = np.max(np.abs(poly(np.linspace(-1, 1, 1000))))
        if max_val > 0.999:
            poly.coef *= (0.999 / max_val)
            delta_eff *= (0.999 / max_val)

        phases = QuantumSignalProcessingPhases(poly, signal_operator="Wx")
        return [float(phi) for phi in phases], delta_eff, sigma

    def _validate_input(self):
        s = np.linalg.svd(self.A, compute_uv=False)
        if np.any(s >= 1 + 1e-10):
            print(f"Warning: Singular values must be < 1")
            return False
        return True

    def get_block_encoding(self):
        N = self.A.shape[0]
        I = np.eye(N)
        A_dag = self.A.conj().T
        
        top_right = scipy.linalg.sqrtm(I - self.A @ A_dag)
        bottom_left = scipy.linalg.sqrtm(I - A_dag @ self.A)
        bottom_right = -A_dag
        
        U_matrix = np.block([[self.A, top_right],
                            [bottom_left, bottom_right]])
        return Operator(U_matrix)

    def apply_projector_phase(self, circuit, phi, qubits):
        circuit.rx(-2 * phi, qubits[0])

    def construct_qsvt_circuit(self):
        q_data = QuantumRegister(self.n, 'b')
        q_anc = QuantumRegister(self.ancilla_qubits, 'anc')
        c = ClassicalRegister(self.n + self.ancilla_qubits, 'meas')
        qc = QuantumCircuit(q_anc, q_data, c)
        
        qc.prepare_state(Statevector(self.b), q_data)
        qc.barrier()

        U_op = self.get_block_encoding()
        U_gate = U_op.to_instruction()
        U_dag_gate = U_op.adjoint().to_instruction()

        for i in range(len(self.angles) - 1):
            self.apply_projector_phase(qc, self.angles[i], q_anc)
            
            if i % 2 == 0:
                qc.append(U_gate, list(q_data) + list(q_anc))
            else:
                qc.append(U_dag_gate, list(q_data) + list(q_anc))
        
        self.apply_projector_phase(qc, self.angles[-1], q_anc)
        qc.barrier()
        qc.measure(range(qc.num_qubits), range(qc.num_qubits))

        print(f"Circuit width: {qc.width()}, depth: {qc.depth()}")
        return qc

    def solve(self, stateVector=True):
        if not self.dataOK:
            return None
        
        qc = self.construct_qsvt_circuit()
        
        if stateVector:
            print("Running QSVT circuit on statevector simulator...")
            qc_no_meas = qc.copy()
            qc_no_meas.remove_final_measurements()
            sv = Statevector.from_instruction(qc_no_meas)
            u_qsvt = sv.data[::2]
        else:
            print("Running QSVT circuit on qasm_simulator...")
            backend = Aer.get_backend('qasm_simulator')
            t_qc = transpile(qc, backend)
            counts = backend.run(t_qc, shots=self.nShots).result().get_counts()
            
            success_counts = {k: v for k, v in counts.items() if k.endswith('0')}
            total_success = sum(success_counts.values())
            
            if total_success == 0:
                return np.zeros(2**self.n)
                
            u_qsvt = np.zeros(2**self.n, dtype=complex)
            for bitstr, count in success_counts.items():
                data_bits = bitstr[:self.n]
                idx = int(data_bits, 2)
                u_qsvt[idx] = np.sqrt(count / total_success)
        
        u_qsvt = u_qsvt * (np.conj(u_qsvt[0]) / np.abs(u_qsvt[0]))
        return u_qsvt / np.linalg.norm(u_qsvt)

if __name__ == "__main__":
    
    example = 1
    if(example == 1):
        print("\n--- Testing 2x2 ---")
        A = np.array([[0.9, 0], [0, 0.1]])  # κ = 9
        b = np.array([1, -1]) / np.sqrt(2)
        kappa = 9

    elif (example == 2):
        print("\n--- Testing 4x4 ---")
        A = np.array([[0.5, -0.2, 0.1, -0.1], 
                  [-0.2, 0.5, -0.1, 0.1],
                  [0.1, -0.1, 0.5, -0.2],
                  [-0.1, 0.1, -0.2, 0.5]]) # κ = 3
        b = np.array([1, 0, 0, 1]) / np.sqrt(2)
        kappa = 3
    target_error = 1e-8
    degree = 50
    for method in ['sunderhauf', 'interpolation']:
        print(f"\nMethod: {method}")
        if (method == 'interpolation'):
            print(f"Using degree: {degree}")
            solver = myQSVT(A, b, degree=degree, kappa=kappa, method=method)
        else:
            print(f"Using target error: {target_error}")
            solver = myQSVT(A, b,kappa=kappa, method=method, target_error=target_error)

        x_qsvt = solver.solve()
 
        x_classical = np.linalg.solve(A, b)
        x_classical /= np.linalg.norm(x_classical)
    
        print(f"QSVT:      {np.round(x_qsvt, 4)}")
        print(f"Classical: {np.round(x_classical, 4)}")
        print(f"Fidelity:  {np.abs(np.vdot(x_qsvt, x_classical))**2:.6f}") 

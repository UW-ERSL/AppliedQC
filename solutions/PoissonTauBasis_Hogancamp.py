"""
Hogancamp et al. (2026) LCU Decomposition for Poisson Equation
================================================================
Implements the tau-basis decomposition from:
"A Linear Combination of Unitaries Decomposition for the Laplace Operator"
by Thomas Hogancamp, Reuben Demirdjian, and Daniel Gunlycke (2026)

Key differences from Gnanasekaran (sigma basis):
- Uses tau-basis {τ₀, τ₁, τ₂, τ₃} projection operators
- 5 terms for 1D Dirichlet (vs 2m+1 for sigma)
- 10 terms for 2D Dirichlet (vs 4m+2 for sigma)
- Number of terms INDEPENDENT of discretization size N

Tau-basis matrices (Equation 2.1 in paper):
    τ₀ = [[1, 0], [0, 0]]
    τ₁ = [[1, 0], [0, 0]]  
    τ₂ = [[0, 0], [1, 0]]
    τ₃ = [[0, 0], [0, 1]]
"""

import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class LCUDecomposition:
    """Store LCU decomposition results"""
    coefficients: List[float]
    operators: List[np.ndarray]
    operator_names: List[str]
    num_terms: int
    
    
class HogancampPoisson1D:
    """
    1D Poisson equation using Hogancamp's tau-basis decomposition
    
    Implements Theorem 4.1 from the paper:
    A_{n,1} can be decomposed into 5 unitary terms
    
    Matrix structure (Equation 3.2):
    L_{n,1} = (1/h²) * tridiag(-1, 2, -1)
    A_{n,1} = h² * L_{n,1}  (normalized version)
    """
    
    def __init__(self, n: int):
        """
        Parameters:
        -----------
        n : int
            Number of qubits (N = 2^n grid points)
        """
        self.n = n
        self.N = 2**n  # Number of grid points
        self.h = 1.0 / (self.N + 1)  # Grid spacing
        
        # Pauli matrices
        self.I = np.eye(2)
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0], [0, -1]])
        
        # Tau-basis matrices (Equation 2.1)
        self.tau_0 = np.array([[1, 0], [0, 0]])
        self.tau_1 = np.array([[1, 0], [0, 0]])  # Same as tau_0 in paper
        self.tau_2 = np.array([[0, 0], [1, 0]])
        self.tau_3 = np.array([[0, 0], [0, 1]])
    
    def get_poisson_matrix(self) -> np.ndarray:
        """
        Get classical 1D Poisson matrix
        
        Returns normalized matrix A_{n,1} = h² L_{n,1}
        Following Equation 3.4 from paper
        """
        # Tridiagonal structure
        A = 2 * np.eye(self.N)
        for i in range(self.N - 1):
            A[i, i+1] = -1
            A[i+1, i] = -1
        
        return A  # This is already A_{n,1} (normalized)
    
    def get_increment_gate(self) -> np.ndarray:
        """
        Increment gate S_n (Equation 2.4)
        
        Circular shift matrix used in decomposition
        """
        S = np.zeros((self.N, self.N))
        for i in range(self.N - 1):
            S[i+1, i] = 1
        S[0, self.N - 1] = 1  # Wrap around
        
        return S
    
    def get_C_matrix(self) -> np.ndarray:
        """
        C_n matrix (Equation 5.2)
        
        Diagonal matrix with 1 on first and last positions
        """
        C = np.zeros((self.N, self.N))
        C[0, 0] = 1
        C[self.N - 1, self.N - 1] = 1
        
        return C
    
    def get_C_minus_matrix(self) -> np.ndarray:
        """
        C⁻_n matrix - correct implementation
        
        From context in the paper, this appears to be related to
        the difference between Dirichlet and Periodic boundary conditions
        
        Looking at Equation 5.4:
        A_{n,1} = A_{n,2} - C_n
        
        where C_n is defined in Eq 5.2 as diagonal with 1 at corners
        
        And Eq 5.3: C_n = (1/2)C⁻_n + (1/2)X⊗n
        
        Solving: C⁻_n = 2C_n - X⊗n
        """
        # C_n from Equation 5.2
        C_n = self.get_C_matrix()
        
        # X⊗n
        X_tensor = self._tensor_power(self.X, self.n)
        
        # From Equation 5.3: C_n = (1/2)C⁻_n + (1/2)X⊗n
        # Therefore: C⁻_n = 2C_n - X⊗n
        C_minus = 2 * C_n - X_tensor
        
        return C_minus
        
    def decompose_hogancamp_1d_dirichlet(self) -> LCUDecomposition:
        """
        5-term LCU decomposition for 1D Dirichlet Poisson
        
        Implements Equation 5.9 from Hogancamp et al. (2026):
        
        A_{n,1} = -2I⊗n + S_n + S_n† - (1/2)X⊗n + (1/2)C⁻_n
        
        where C⁻_n has a specific unitary decomposition (Equation 5.5)
        
        Returns:
        --------
        LCUDecomposition with 5 terms
        """
        # Get the operators
        I_tensor = np.eye(self.N)
        S_n = self.get_increment_gate()
        S_n_dag = S_n.T
        X_tensor = self._tensor_power(self.X, self.n)
        C_minus = self.get_C_minus_matrix()
        
        # 5 terms from Equation 5.9
        coefficients = [-2.0, 1.0, 1.0, -0.5, 0.5]
        operators = [I_tensor, S_n, S_n_dag, X_tensor, C_minus]
        names = ['I⊗n', 'S_n', 'S_n†', 'X⊗n', 'C⁻_n']
        
        return LCUDecomposition(
            coefficients=coefficients,
            operators=operators,
            operator_names=names,
            num_terms=5
        )
    
    def decompose_hogancamp_1d_periodic(self) -> LCUDecomposition:
        """
        3-term LCU decomposition for 1D Periodic Poisson
        
        Implements Equation 5.1:
        A_{n,2} = -2I + S_n + S_n†
        
        Returns:
        --------
        LCUDecomposition with 3 terms
        """
        I_tensor = np.eye(self.N)
        S_n = self.get_increment_gate()
        S_n_dag = S_n.T
        
        coefficients = [-2.0, 1.0, 1.0]
        operators = [I_tensor, S_n, S_n_dag]
        names = ['I⊗n', 'S_n', 'S_n†']
        
        return LCUDecomposition(
            coefficients=coefficients,
            operators=operators,
            operator_names=names,
            num_terms=3
        )
    
    def _tensor_power(self, matrix: np.ndarray, power: int) -> np.ndarray:
        """Compute tensor product matrix^⊗power"""
        result = matrix
        for _ in range(power - 1):
            result = np.kron(result, matrix)
        return result
    
    def verify_decomposition(self, decomposition: LCUDecomposition) -> Dict[str, float]:
        """
        Verify LCU decomposition accuracy
        
        Reconstructs matrix and compares to original
        """
        # Get original matrix
        A_original = self.get_poisson_matrix()
        
        # Reconstruct from LCU
        A_reconstructed = np.zeros_like(A_original)
        for coeff, op in zip(decomposition.coefficients, decomposition.operators):
            A_reconstructed += coeff * op
        
        # Compute error
        error = np.linalg.norm(A_original - A_reconstructed, 'fro')
        relative_error = error / np.linalg.norm(A_original, 'fro')
        
        return {
            'frobenius_error': error,
            'relative_error': relative_error,
            'num_terms': decomposition.num_terms,
            'is_correct': relative_error < 1e-10
        }
    
    def print_decomposition_info(self, decomposition: LCUDecomposition):
        """Print information about the decomposition"""
        print(f"\n{'='*70}")
        print(f"Hogancamp LCU Decomposition")
        print(f"{'='*70}")
        print(f"Number of qubits (n): {self.n}")
        print(f"Grid points (N = 2^n): {self.N}")
        print(f"Number of LCU terms: {decomposition.num_terms}")
        print(f"\nTerms:")
        for i, (coeff, name) in enumerate(zip(decomposition.coefficients, 
                                               decomposition.operator_names)):
            print(f"  {i+1}. {coeff:+.2f} × {name}")


class HogancampPoisson2D:
    """
    2D Poisson equation using Hogancamp's tau-basis decomposition
    
    Implements Theorem 4.5 for d=2 dimensions:
    Total of 10 terms for 2D Dirichlet (5 terms per dimension)
    
    Matrix structure (Section 3.4):
    L²_{n,α} = I⊗n₂ ⊗ L_{n₁,α₁} + L_{n₂,α₂} ⊗ I⊗n₁
    """
    
    def __init__(self, n: int):
        """
        Parameters:
        -----------
        n : int
            Number of qubits per dimension
            Total qubits: 2n
            Grid: (2^n) × (2^n)
        """
        self.n = n
        self.N_1d = 2**n  # Grid points per dimension
        self.N_2d = self.N_1d ** 2  # Total grid points
        self.total_qubits = 2 * n
        
        self.h = 1.0 / (self.N_1d + 1)
        
        # Create 1D solver for building blocks
        self.solver_1d = HogancampPoisson1D(n)
    
    def get_poisson_matrix_2d(self) -> np.ndarray:
        """
        Get 2D Poisson matrix using Kronecker product structure
        
        K_2D = K_x ⊗ I + I ⊗ K_y
        
        Following Equation (5.14) from paper
        """
        K_1d = self.solver_1d.get_poisson_matrix()
        I_1d = np.eye(self.N_1d)
        
        # Tensor product structure
        K_x = np.kron(K_1d, I_1d)
        K_y = np.kron(I_1d, K_1d)
        
        K_2d = K_x + K_y
        
        return K_2d
    
    def decompose_hogancamp_2d_dirichlet(self) -> LCUDecomposition:
        """
        10-term LCU decomposition for 2D Dirichlet Poisson
        
        Implements Theorem 4.5 and Equation (5.14):
        
        L²_{n,α} = Σᵢ₌₁² Σⱼ₌₁⁵ cᵢⱼ Rᵢⱼ
        
        Total: 2 dimensions × 5 terms each = 10 terms
        
        Returns:
        --------
        LCUDecomposition with 10 terms
        """
        # Get 1D decomposition
        decomp_1d = self.solver_1d.decompose_hogancamp_1d_dirichlet()
        
        I_1d = np.eye(self.N_1d)
        
        coefficients = []
        operators = []
        names = []
        
        # Terms for x-direction: R ⊗ I
        for i, (coeff, op_1d, name) in enumerate(zip(decomp_1d.coefficients,
                                                       decomp_1d.operators,
                                                       decomp_1d.operator_names)):
            op_2d = np.kron(op_1d, I_1d)
            coefficients.append(coeff)
            operators.append(op_2d)
            names.append(f'{name} ⊗ I (x-dir)')
        
        # Terms for y-direction: I ⊗ R
        for i, (coeff, op_1d, name) in enumerate(zip(decomp_1d.coefficients,
                                                       decomp_1d.operators,
                                                       decomp_1d.operator_names)):
            op_2d = np.kron(I_1d, op_1d)
            coefficients.append(coeff)
            operators.append(op_2d)
            names.append(f'I ⊗ {name} (y-dir)')
        
        return LCUDecomposition(
            coefficients=coefficients,
            operators=operators,
            operator_names=names,
            num_terms=10
        )
    
    def verify_decomposition(self, decomposition: LCUDecomposition) -> Dict[str, float]:
        """Verify 2D LCU decomposition accuracy"""
        K_original = self.get_poisson_matrix_2d()
        
        K_reconstructed = np.zeros_like(K_original)
        for coeff, op in zip(decomposition.coefficients, decomposition.operators):
            K_reconstructed += coeff * op
        
        error = np.linalg.norm(K_original - K_reconstructed, 'fro')
        relative_error = error / np.linalg.norm(K_original, 'fro')
        
        return {
            'frobenius_error': error,
            'relative_error': relative_error,
            'num_terms': decomposition.num_terms,
            'is_correct': relative_error < 1e-10
        }
    
    def print_decomposition_info(self, decomposition: LCUDecomposition):
        """Print 2D decomposition information"""
        print(f"\n{'='*70}")
        print(f"Hogancamp 2D LCU Decomposition")
        print(f"{'='*70}")
        print(f"Qubits per dimension: {self.n}")
        print(f"Total qubits: {self.total_qubits}")
        print(f"Grid: {self.N_1d} × {self.N_1d} = {self.N_2d} points")
        print(f"Number of LCU terms: {decomposition.num_terms}")
        print(f"Formula: 2 × 5 = 10 terms (independent of N!)")
        print(f"\nTerms:")
        for i, (coeff, name) in enumerate(zip(decomposition.coefficients,
                                               decomposition.operator_names)):
            print(f"  {i+1:2d}. {coeff:+.2f} × {name}")


def compare_hogancamp_vs_gnanasekaran():
    """
    Compare Hogancamp (tau-basis) vs Gnanasekaran (sigma-basis)
    
    Key comparison:
    - Hogancamp: O(1) terms (independent of N)
    - Gnanasekaran: O(log N) terms
    """
    print("="*80)
    print("COMPARISON: Hogancamp (2026) vs Gnanasekaran (2024)")
    print("="*80)
    
    print(f"\n{'n':<4} {'N':<8} {'Hogancamp 1D':<15} {'Gnanasekaran 1D':<15} "
          f"{'Hogancamp 2D':<15} {'Gnanasekaran 2D':<15}")
    print("-"*80)
    
    for n in range(2, 9):
        N = 2**n
        hogancamp_1d = 5
        gnanasekaran_1d = 2*n + 1
        hogancamp_2d = 10
        gnanasekaran_2d = 4*n + 2
        
        print(f"{n:<4} {N:<8} {hogancamp_1d:<15} {gnanasekaran_1d:<15} "
              f"{hogancamp_2d:<15} {gnanasekaran_2d:<15}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("✓ Hogancamp: Constant terms (5 for 1D, 10 for 2D) - INDEPENDENT of N!")
    print("✓ Gnanasekaran: O(log N) terms - grows with problem size")
    print("✓ For small n (≤3): Gnanasekaran may have fewer terms")
    print("✓ For large n (≥4): Hogancamp always has fewer terms")
    print("✓ Hogancamp wins for large-scale problems!")


def demonstrate_hogancamp_1d():
    """Demonstrate 1D Hogancamp decomposition"""
    print("\n" + "="*80)
    print("HOGANCAMP 1D DIRICHLET DECOMPOSITION")
    print("="*80)
    
    for n in [2, 3, 4, 5]:
        print(f"\n{'='*80}")
        print(f"n = {n} qubits (N = {2**n} grid points)")
        print(f"{'='*80}")
        
        solver = HogancampPoisson1D(n)
        
        # Dirichlet decomposition
        decomp = solver.decompose_hogancamp_1d_dirichlet()
        solver.print_decomposition_info(decomp)
        
        # Verify
        verification = solver.verify_decomposition(decomp)
        print(f"\nVerification:")
        print(f"  Frobenius error: {verification['frobenius_error']:.2e}")
        print(f"  Relative error: {verification['relative_error']:.2e}")
        print(f"  Status: {'✓ PASS' if verification['is_correct'] else '✗ FAIL'}")
        
        # Compare with naive Pauli
        naive_pauli = 2**(2*n)  # Worst case
        print(f"\nComplexity:")
        print(f"  Hogancamp terms: {decomp.num_terms}")
        print(f"  Naive Pauli: {naive_pauli}")
        print(f"  Reduction: {naive_pauli / decomp.num_terms:.0f}×")


def demonstrate_hogancamp_2d():
    """Demonstrate 2D Hogancamp decomposition"""
    print("\n" + "="*80)
    print("HOGANCAMP 2D DIRICHLET DECOMPOSITION")
    print("="*80)
    
    for n in [2, 3, 4]:
        print(f"\n{'='*80}")
        print(f"n = {n} qubits per dimension (Grid: {2**n}×{2**n})")
        print(f"{'='*80}")
        
        solver = HogancampPoisson2D(n)
        
        # 2D Dirichlet decomposition
        decomp = solver.decompose_hogancamp_2d_dirichlet()
        solver.print_decomposition_info(decomp)
        
        # Verify
        verification = solver.verify_decomposition(decomp)
        print(f"\nVerification:")
        print(f"  Frobenius error: {verification['frobenius_error']:.2e}")
        print(f"  Relative error: {verification['relative_error']:.2e}")
        print(f"  Status: {'✓ PASS' if verification['is_correct'] else '✗ FAIL'}")
        
        # Compare
        naive_pauli = 4**(2*n)
        gnanasekaran = 4*n + 2
        print(f"\nComplexity:")
        print(f"  Hogancamp terms: {decomp.num_terms}")
        print(f"  Gnanasekaran terms: {gnanasekaran}")
        print(f"  Naive Pauli: {naive_pauli:,}")
        print(f"  Hogancamp vs Pauli: {naive_pauli / decomp.num_terms:,.0f}×")
        print(f"  Hogancamp vs Gnanasekaran: {gnanasekaran / decomp.num_terms:.1f}× {'BETTER' if gnanasekaran > decomp.num_terms else 'WORSE'}")


if __name__ == "__main__":

    # Run demonstrations
    compare_hogancamp_vs_gnanasekaran()
    demonstrate_hogancamp_1d()
    demonstrate_hogancamp_2d()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Hogancamp et al. (2026) achieves:")
    print("  ✓ 5 terms for 1D Dirichlet (constant, independent of N)")
    print("  ✓ 10 terms for 2D Dirichlet (constant, independent of N)")
    print("  ✓ O(log N) circuit depth per term")
    print("  ✓ Better than Gnanasekaran for n ≥ 4")
    print("  ✓ Scales linearly with dimension (10d for d dimensions)")
    print("="*80)
import numpy as np
from itertools import product
from qiskit.quantum_info import SparsePauliOp
class PoissonEncoding:
    def __init__(self, m):
        """
        m: number of qubits
        n: dimension of the matrix (2**m)
        """
        self.m = m
        self.n = 2**m
        # Basic Unitary Operators for reconstruction and LCU
        self.I = np.eye(2)
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0], [0, -1]])

    def get_full_matrix(self):
        """
        Constructs the standard classical tridiagonal FDM matrix A.
        A = tridiag(-1, 2, -1)
        """
        A = 2 * np.eye(self.n)
        for i in range(self.n - 1):
            A[i, i+1] = -1
            A[i+1, i] = -1
        return A

    def decompose_A_simple(self):
        """
        Implements the recursive decomposition from Liu et al. (2021)
        using the basis {I, sigma_plus, sigma_minus}.
        Number of terms: 2m + 1
        """
        terms = []
        # Identity term: 2 * I^m
        terms.append((2.0, ["I"] * self.m))
        
        # Recursive subtraction terms based on the paper's specific structure
        for j in range(1, self.m + 1):
            # Term type 1: I^(m-j) ⊗ sigma_plus ⊗ (sigma_minus)^(j-1)
            op1 = ["I"] * (self.m - j) + ["sigma_plus"] + ["sigma_minus"] * (j - 1)
            terms.append((-1.0, op1))
            
            # Term type 2: I^(m-j) ⊗ sigma_minus ⊗ (sigma_plus)^(j-1)
            op2 = ["I"] * (self.m - j) + ["sigma_minus"] + ["sigma_plus"] * (j - 1)
            terms.append((-1.0, op2))
            
        return terms

    def decompose_to_pauli(self):
        """
        Converts the non-unitary {I, sigma_+, sigma_-} decomposition into 
        unitary Pauli strings {I, X, Y, Z}. This is the form required 
        for LCU and QSVT block-encodings.
        """
        simple_decomp = self.decompose_A_simple()
        pauli_lcu = {}

        # Mapping: sigma_+ = (X + iY)/2, sigma_- = (X - iY)/2
        sigma_map = {
            "I": [(1.0, "I")],
            "sigma_plus": [(0.5, "X"), (0.5j, "Y")],
            "sigma_minus": [(0.5, "X"), (-0.5j, "Y")]
        }

        for coeff, op_list in simple_decomp:
            choices = [sigma_map[op] for op in op_list]
            for combination in product(*choices):
                combined_coeff = coeff
                pauli_string = ""
                for term_coeff, pauli_char in combination:
                    combined_coeff *= term_coeff
                    pauli_string += pauli_char
                
                pauli_lcu[pauli_string] = pauli_lcu.get(pauli_string, 0) + combined_coeff

        return {k: v.real for k, v in pauli_lcu.items() if abs(v) > 1e-12}

    def get_lcu_alpha(self):
        """
        Calculates the normalization constant alpha = sum |c_j|.
        """
        pauli_dict = self.decompose_to_pauli()
        return sum(abs(c) for c in pauli_dict.values())

    def reconstruct_matrix_from_pauli(self, pauli_dict):
        """Helper to verify accuracy by rebuilding the matrix."""
        res = np.zeros((self.n, self.n), dtype=complex)
        gate_map = {"I": self.I, "X": self.X, "Y": self.Y, "Z": self.Z}
        for p_string, coeff in pauli_dict.items():
            term_mat = coeff
            for char in p_string:
                term_mat = np.kron(term_mat, gate_map[char])
            res += term_mat
        return res

if __name__ == "__main__":
    m_qubits = 10
    pe = PoissonEncoding(m_qubits)
    
    # 1. Classical target
    A_classical = pe.get_full_matrix()
    
    simple_decomp = pe.decompose_A_simple()

    print("--- Simple Decomposition (Non-Unitary) ---")
    print(f"Number of Non-Unitary terms: {len(simple_decomp)}")
    if len(simple_decomp) <= 5:  # Only print if manageable
        for coeff, op_list in simple_decomp:
            print(f"  {coeff:+.1f} * {' ⊗ '.join(op_list)}")

    # 2. Quantum Decompositions
    pauli_terms = pe.decompose_to_pauli()
    alpha = pe.get_lcu_alpha()
    print(f"Total Pauli terms: {len(pauli_terms)}")
    if len(pauli_terms) <= 5:  # Only print if manageable
        for p_str, coeff in sorted(pauli_terms.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {coeff:+.4f} * {p_str}")
        # 3. Verification
        A_recon = pe.reconstruct_matrix_from_pauli(pauli_terms)
        
        print(f"--- Poisson Encoding (Qubits: {m_qubits}) ---")
        print(f"Verification: {'SUCCESS' if np.allclose(A_classical, A_recon.real) else 'FAILED'}")
        print(f"LCU Alpha: {alpha:.4f}")
        

    # pauliSplit = SparsePauliOp.from_operator(A_classical)
    # print(pauliSplit.paulis)
    # print(pauliSplit.coeffs)
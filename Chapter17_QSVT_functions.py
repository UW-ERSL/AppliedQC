import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import Estimator
from qiskit_aer import Aer
from qiskit.quantum_info import SparsePauliOp
from IPython.display import display

from qiskit_algorithms.optimizers import ADAM  # Yes, there's a Qiskit ADAM!

import numpy as np
import scipy
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import Aer

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
# Assuming your library is pyqsp or similar
from pyqsp.angle_sequence import QuantumSignalProcessingPhases 

def get_inverse_phases(degree, kappa = 2):
    """
    Generates QSVT phase angles for f(x) = 1/x approximation.
    
    Args:
        degree (int): The polynomial degree (higher = more accurate).
        kappa (float): Condition number (1/delta).
        solver_func: Your QuantumSignalProcessingPhases class/function.
    """
    # Ensure degree is odd (QSVT for 1/x requires odd degree)
    if degree % 2 == 0:
        degree += 1
        print(f"Degree must be odd for QSVT 1/x. Using degree={degree}.")
    if kappa <= 1:
        kappa = 2
        print(f"kappa must be > 1. Using kappa={kappa}.")
    delta = 1/kappa
    
    # 1. Define the target function (scaled to stay <= 1)
    # We use a factor (e.g., delta) to ensure f(x) fits in the QSVT box.
    scale_factor = delta
    def target_func(x):
        # Handle the region near zero to avoid singularities during fitting
        return scale_factor / np.where(np.abs(x) < delta, delta, x)

    # 2. Generate Chebyshev coefficients via interpolation
    # Domain is [-1, 1], but we care about [delta, 1]
    poly = Chebyshev.interpolate(target_func, deg=degree, domain=[-1, 1])
    # we want only odd powers for 1/x approximation
    poly.coef[::2] = 0 # This kills the even terms (0, 2, 4...
    print(f"Chebyshev coefficients (odd terms only): {poly.coef}")
    # 3. Compute Phase Angles
    # Note: solver_func is your QuantumSignalProcessingPhases
    phases = QuantumSignalProcessingPhases(poly,  signal_operator="Wz" )
    print(phases)
    return [float(phi) for phi in phases], scale_factor


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
        self.angles, self.tau  = get_inverse_phases(degree,kappa)
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
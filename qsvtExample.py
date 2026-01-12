import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Operator
from qiskit.circuit.library import UnitaryGate
from IPython.display import display

import matplotlib.pyplot as plt
def get_block_encoding(matrix):
    """
    Simplistic block encoding of a 2x2 matrix A into a 4x4 Unitary.
    A must be scaled such that its singular values are <= 1.
    """
    # Scale matrix for block encoding
    norm_a = np.linalg.norm(matrix, 2)
    A_scaled = matrix / norm_a if norm_a > 1 else matrix
    
    dim = A_scaled.shape[0]
    # Create a unitary U = [[A, sqrt(I-AA*)] , [sqrt(I-A*A), -A*]]
    # For a 2x2, this creates a 4x4 (2 qubits)
    top_left = A_scaled
    top_right = np.sqrt(np.eye(dim) - A_scaled @ A_scaled.T.conj())
    bot_left = np.sqrt(np.eye(dim) - A_scaled.T.conj() @ A_scaled)
    bot_right = -A_scaled.T.conj()
    
    U = np.block([
        [top_left, top_right],
        [bot_left, bot_right]
    ])
    return UnitaryGate(U, label="U_A"), norm_a

def apply_qsvt_inversion(matrix, b_vector):
    # 1. Setup Qu-registers
    # Ancilla for QSP/QSVT phases, System for the matrix/vector
    ancilla = QuantumRegister(1, name='ancilla')
    system = QuantumRegister(1, name='system')
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(ancilla, system, cr)

    # 2. Prepare state |b>
    # Normalize b and initialize system qubit
    b_norm = b_vector / np.linalg.norm(b_vector)
    if b_norm[0] < 0: b_norm *= -1 # Global phase fix
    qc.initialize(b_norm, system)
    
    # 3. Define Block Encoding and Phases
    U_A, scale = get_block_encoding(matrix)
    
    # Sample phase angles for 1/x approximation (Degree 5)
    # In a real scenario, these are calculated via PyQSP
    phases = [0.52, -1.25, 2.1, -1.25, 0.52] 
    
    # 4. QSVT Iteration Loop
    # The structure: Rz(phi_0) -> [U -> Rz(phi_i) -> U* -> Rz(phi_i+1)]
    qc.h(ancilla) 
    
    for i, phi in enumerate(phases):
        # Controlled Rotation (Phase shift)
        qc.crz(phi, ancilla[0], system[0])
        
        # Apply Block Encoding U_A
        if i % 2 == 0:
            qc.append(U_A, [ancilla[0], system[0]])
        else:
            qc.append(U_A.inverse(), [ancilla[0], system[0]])
            
    qc.h(ancilla)
    
    # 5. Measurement (Success if ancilla is |0>)
    qc.measure(ancilla, cr)
    return qc

def get_random_unitary(num_qubits, seed=4):
    np.random.seed(seed)
    X = np.random.rand(2**num_qubits, 2**num_qubits)
    U, s, V = np.linalg.svd(X)
    return U @ V.T

REG_SIZE = 2
A = get_random_unitary(REG_SIZE)

# Example 2x2 matrix

A = get_random_unitary(1)
eigenvalues = np.linalg.eigvals(A)
print("Eigenvalues of A:", eigenvalues)
print("Magnitudes of eigenvalues:", np.abs(eigenvalues))

# Example Usage

b = np.array([1, 0])

qsvt_circuit = apply_qsvt_inversion(A, b)
print("QSVT Matrix Inversion Circuit Depth:", qsvt_circuit.depth())

qsvt_circuit.draw('mpl')
plt.show()
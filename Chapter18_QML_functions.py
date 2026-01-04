


import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import Estimator
from qiskit_aer import Aer
from qiskit.quantum_info import SparsePauliOp
from IPython.display import display

from qiskit_algorithms.optimizers import ADAM  # Yes, there's a Qiskit ADAM!

# Compute gradients via parameter-shift rule (Qiskit supports this)
from qiskit.circuit import ParameterVector



def create_parametric_circuit(n_qubits, n_layers):
    """Create a parameterized quantum circuit (PQC) with angle encoding and variational layers."""
    # Parameters
    x = ParameterVector("x", n_qubits)
    theta = ParameterVector("θ", n_layers * n_qubits * 3)

    qc = QuantumCircuit(n_qubits)

    # --- Angle encoding ---
    for i in range(n_qubits):
        qc.ry(x[i], i)

    # --- Variational layers ---
    k = 0
    for _ in range(n_layers):
        for i in range(n_qubits):
            qc.rx(theta[k], i); k += 1
            qc.ry(theta[k], i); k += 1
            qc.rz(theta[k], i); k += 1
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)

    return qc

def pqc_predict_batch(theta_vec, X, qc, observable, estimator):
    """
    theta_vec: shape (len(theta),)
    X: shape (batch_size, n_qubits)
    returns: predictions shape (batch_size,)

    This evaluates the PQC for many x values at once (faster than looping).
    """
    param_values_batch = [
        np.concatenate([x_val, theta_vec]) for x_val in X
    ]
    job = estimator.run([(qc, observable, param_values_batch)])
    res = job.result()
    evs = res[0].data.evs  # numpy array of expvals
    return np.array(evs, dtype=float)



def huber_loss(preds, y, delta=0.2):
    r = preds - y
    abs_r = np.abs(r)
    quad = np.minimum(abs_r, delta)
    lin = abs_r - quad
    return np.mean(0.5*quad**2 + delta*lin)

def cosine_fidelity(y_hat, y):
    y_hat = np.asarray(y_hat, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = (np.linalg.norm(y_hat) * np.linalg.norm(y) + 1e-12)
    return float(np.dot(y_hat, y) / denom)

def cosine_fidelity_01(y_hat, y):
    return 0.5 * (1.0 + cosine_fidelity(y_hat, y))

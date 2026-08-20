"""
Quantum test circuits for Chapter 10, "Quantum Tests".

Two helpers used by the chapter: the RY/RZ rotation matrices from which the
combined unitary U = U_phi^dag U_psi of the Hadamard test is built, and a
multi-qubit inner-product estimator that applies a controlled-SWAP between
each corresponding pair of qubits in two registers.

The Hadamard-test and swap-test circuits themselves are built inline in the
listings and in the notebook.

Reference
---------
Buhrman, Cleve, Watrous and de Wolf (2001), "Quantum fingerprinting",
Phys. Rev. Lett. 87, 167902 --- the swap test.
"""

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate

from Chapter08_QuantumGates_functions import simulate_measurements #type: ignore

def ryMatrix(alpha):
    """
    Return the 2x2 rotation matrix R_y(α) about the Bloch-sphere y-axis.

    R_y(α) = [[cos(α/2), -sin(α/2)], [sin(α/2), cos(α/2)]].

    Parameters
    ----------
    alpha : float
        Rotation angle α (radians).

    Returns
    -------
    numpy.ndarray
        Real (2, 2) unitary rotation matrix.
    """
    return np.array([[np.cos(alpha/2), -np.sin(alpha/2)], [np.sin(alpha/2), np.cos(alpha/2)]])

def rzMatrix(omega):
    """
    Return the 2x2 rotation matrix R_z(ω) about the Bloch-sphere z-axis.

    R_z(ω) = diag(e^{-iω/2}, e^{+iω/2}), a diagonal phase rotation.

    Parameters
    ----------
    omega : float
        Rotation angle ω (radians).

    Returns
    -------
    numpy.ndarray
        Complex (2, 2) diagonal unitary rotation matrix.
    """
    return np.array([[np.exp(-1j*omega/2), 0], [0, np.exp(1j*omega/2)]])

def innerProductEstimation(ua, ub, shots = 10000):
    """
    Estimate |<ua|ub>| using  Controlled Swap.
    
    Parameters
    ----------
    ua : array
        State |a> vector
    ub : array
        State |b> vector
    shots : int
        Measurement shots
    backend : Backend, optional
        Quantum backend
    
    Returns
    -------
    float
        Estimated |<ua|ub>|
    """
    # --- 2. Circuit Setup ---
    num_data_qubits = int(np.log2(len(ua)))
    q_aux = QuantumRegister(1, 'anc')
    q_psi = QuantumRegister(num_data_qubits, 'A')
    q_phi = QuantumRegister(num_data_qubits, 'B')
    c_res = ClassicalRegister(1, 'm')

     # Build circuit
    qc = QuantumCircuit(q_aux, q_psi, q_phi, c_res)

    # Create a SUB-CIRCUIT just for the state preparation
    sub_a = QuantumCircuit(num_data_qubits)
    sub_a.prepare_state(ua, range(num_data_qubits))
    # Turn the sub-circuit into a single gate and give it a label
    state_a_gate = sub_a.to_gate(label="A")
    qc.append(state_a_gate, q_psi)

    sub_b = QuantumCircuit(num_data_qubits)
    sub_b.prepare_state(ub, range(num_data_qubits))
    state_b_gate = sub_b.to_gate(label="B")
    qc.append(state_b_gate, q_phi)

    qc.barrier()

    qc.h(q_aux[0])
    for i in range(num_data_qubits):
        qc.cswap(q_aux[0], q_psi[i], q_phi[i])
    qc.h(q_aux[0])

    qc.measure(q_aux, c_res)
    # Simulate circuit
    counts = simulate_measurements(qc, shots= shots )
    
    # Calculate |<s|x>|^2
    p0 = counts.get('0', 0) / shots
    overlap_squared = max(0, 2 * p0 - 1)
    innerProduct = np.sqrt(overlap_squared)
    return innerProduct, qc

def innerProductCircuit(alphaPhi, alphaPsi):
    """
    Build the Hadamard-test circuit that estimates Re <phi|psi> for the two
    single-qubit states

        |phi> = R_y(alphaPhi) |0>,     |psi> = R_y(alphaPsi) |0>.

    Setting U = U_phi^dag U_psi turns the overlap into Re <0|U|0>, which the
    Hadamard test recovers from the ancilla statistics as 2 P(0) - 1.

    This is the circuit of Listing 10.1, factored out of the notebook so that
    the *same* circuit object can be handed to the Aer simulator and to IBM
    Quantum hardware without any risk of the two runs drifting apart.

    Parameters
    ----------
    alphaPhi, alphaPsi : float
        R_y rotation angles (radians) preparing |phi> and |psi>.

    Returns
    -------
    QuantumCircuit
        Two qubits, one classical bit.  The classical register keeps Qiskit's
        default name 'c', which is what runCircuitOnIBMQuantum reads back from
        the primitive result (pub_result.data.c).

    Notes
    -----
    For real angles the overlap is real, so the real Hadamard test is enough;
    Section 10.2 shows the sdg variant needed for the imaginary component.
    """
    UMatrix = np.matmul(ryMatrix(alphaPhi).conj().T, ryMatrix(alphaPsi))

    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    U = UnitaryGate(UMatrix, 'U')
    circuit.append(U.control(1), [0, 1])
    circuit.h(0)
    circuit.measure(0, 0)
    return circuit


def innerProductFromCounts(counts, shots=None):
    """
    Convert Hadamard-test ancilla counts into the inner-product estimate.

    The estimator is d = 2 P(0) - 1.  Because P(0) is a binomial proportion,
    its standard error is sqrt(P(0)(1 - P(0))/shots), and the factor of 2
    carries straight through to d.  Reporting this band matters once the same
    circuit is run on hardware: it separates the O(1/sqrt(shots)) statistical
    scatter, which more shots will shrink, from the systematic device bias,
    which they will not.

    Parameters
    ----------
    counts : dict
        Ancilla measurement counts, e.g. {'0': 9330, '1': 670}.
    shots : int, optional
        Total shots.  Defaults to the sum of `counts`, which is the safe choice
        on hardware where the returned total need not equal the requested shots.

    Returns
    -------
    (float, float)
        The estimate 2 P(0) - 1 and its one-sigma shot-noise band.
    """
    if shots is None:
        shots = sum(counts.values())
    if shots == 0:
        raise ValueError("No shots recorded")

    p0 = counts.get('0', 0) / shots
    estimate = 2.0 * p0 - 1.0
    sigma = 2.0 * np.sqrt(max(p0 * (1.0 - p0), 0.0) / shots)
    return estimate, sigma
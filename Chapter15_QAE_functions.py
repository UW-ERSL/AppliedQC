import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit.circuit.library import GroverOperator
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import SparsePauliOp
from qiskit import  transpile
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


from Chapter08_QuantumGates_functions import (simulate_statevector, simulate_measurements, runCircuitOnIBMQuantum,
                                              findActualHardwareRequirements, plot_measurement_results)

from Chapter14_MatrixEncoding_functions import LCU_Ax


def disk_predicate(i0, j0, r):
    return lambda i, j: (i - i0)**2 + (j - j0)**2 <= r*r

def annulus_predicate(i0, j0, r1, r2):
    return lambda i, j: r1*r1 <= (i - i0)**2 + (j - j0)**2 <= r2*r2

def square_with_hole_predicate(i0, j0, r):
    return lambda i, j: not ((i - i0)**2 + (j - j0)**2 <= r*r)

def region_area(m, predicate):
    N = 1 << m
    return sum(predicate(i, j) for i in range(N) for j in range(N))


# ---- Geometry-agnostic black-box oracle -------------------------------------
def region_oracle(m, predicate):
    """
    Mark grid cells (i, j) on a 2^m x 2^m grid where predicate(i, j) is True.
    Registers: flag (1 qubit, TOP wire), x (column i), y (row j).

    Demo oracle: evaluates `predicate` classically and marks matching basis
    states with multi-controlled flips. Exact and works for ANY predicate, but
    its gate count grows with the number of marked cells -- adequate for the
    sampling-cost argument here, not an efficient oracle.
    """
    flag = QuantumRegister(1, "flag")
    x    = QuantumRegister(m, "x")
    y    = QuantumRegister(m, "y")
    qc   = QuantumCircuit(flag, x, y)

    N = 1 << m
    for i in range(N):
        for j in range(N):
            if not predicate(i, j):
                continue
            bits  = [(x[b], (i >> b) & 1) for b in range(m)] + \
                    [(y[b], (j >> b) & 1) for b in range(m)]
            zeros = [q for (q, bit) in bits if bit == 0]
            for q in zeros:                       # target pattern -> all-ones
                qc.x(q)
            qc.mcx([q for (q, _) in bits], flag[0])
            for q in zeros:                       # uncompute
                qc.x(q)
    return qc, (x, y, flag)

# ---- Geometry wrappers: each shape is ONE predicate -------------------------
def disk(m, i0, j0, r):
    return region_oracle(m, disk_predicate(i0, j0, r))

def annulus(m, i0, j0, r_in, r_out):
    return region_oracle(m, annulus_predicate(i0, j0, r_in, r_out))

def square_with_hole(m, i0, j0, r):
    return region_oracle(m, square_with_hole_predicate(i0, j0, r))


# ---- Naive estimation: prepare, measure the flag, count ---------------------
def estimate_area(oracle, regs, shots):
    """
    Quantum-assisted Monte Carlo. Build the uniform superposition over all
    N^2 cells, apply the oracle, measure ONLY the flag, and return the
    fraction of |1> outcomes -- which equals the area fraction.
    """
    x, y, flag = regs
    c = ClassicalRegister(1, "c")
    circ = QuantumCircuit(flag, x, y, c)
    circ.h(x)
    circ.h(y)                                  
    circ.compose(oracle, inplace=True)
    circ.measure(flag[0], c[0])
    
    counts = simulate_measurements(circ, shots=shots)  
    ones   = counts.get('1', 0)
    return ones / shots, counts, circ


def make_measured_circuit(oracle, regs):
    x, y, flag = regs
    c = ClassicalRegister(1, "c")
    circ = QuantumCircuit(flag, x, y, c)
    circ.h(x); circ.h(y)
    circ.compose(oracle, inplace=True)
    circ.measure(flag[0], c[0])
    return circ

def sampling_scaling(oracle, regs, shot_list, a_true, trials=20):
    sim  = AerSimulator()
    circ = make_measured_circuit(oracle, regs).decompose(reps=3)
    tcirc = transpile(circ, sim)                 # transpile ONCE

    rows, rms_errors, predicted = [], [], []
    for shots in shot_list:
        errs = []
        for _ in range(trials):
            counts = sim.run(tcirc, shots=shots).result().get_counts()
            a_hat  = counts.get('1', 0) / shots
            errs.append(a_hat - a_true)
        rms  = np.sqrt(np.mean(np.square(errs)))
        pred = np.sqrt(a_true * (1 - a_true) / shots)
        rows.append((shots, rms, pred)); rms_errors.append(rms); predicted.append(pred)

    return rows, rms_errors, predicted


def weighted_region_circuit(m, predicate, weight_fn, weight_max=None):
    """
    Build a circuit that amplitude-encodes a coordinate weight over a region.

    Registers: wt (weight ancilla, TOP wire), flag (region membership),
    x, y (coordinates).
    For each cell (i,j) with predicate True, the wt ancilla is rotated so that
        P(wt = 1 | cell (i,j)) = h(i,j) / weight_max   in [0, 1].
    On the uniform input, the JOINT probability P(flag=1 AND wt=1) equals
        (1/N^2) * sum_{inside} h(i,j) / weight_max  =  <h * 1_region> / weight_max.

    weight_fn : h(i, j) -> real >= 0   (the quantity being averaged)
    weight_max: scale so h/weight_max lies in [0,1]; if None, computed from the
                max of h over inside cells.
    """
    N = 1 << m
    inside = [(i, j) for i in range(N) for j in range(N) if predicate(i, j)]

    if weight_max is None:
        weight_max = max((weight_fn(i, j) for (i, j) in inside), default=1.0)
        weight_max = weight_max if weight_max > 0 else 1.0

    wt   = QuantumRegister(1, "wt")
    flag = QuantumRegister(1, "flag")
    x    = QuantumRegister(m, "x")
    y    = QuantumRegister(m, "y")
    qc   = QuantumCircuit(wt, flag, x, y)

    for (i, j) in inside:
        ctrl_qubits = list(x) + list(y)
        pattern     = [(i >> b) & 1 for b in range(m)] + [(j >> b) & 1 for b in range(m)]
        zeros = [q for q, bit in zip(ctrl_qubits, pattern) if bit == 0]

        for q in zeros:
            qc.x(q)
        # (1) flag this cell as inside
        qc.mcx(ctrl_qubits, flag[0])
        # (2) rotate the weight ancilla by the cell's weight, controlled on the cell
        frac  = weight_fn(i, j) / weight_max          # in [0, 1]
        theta = 2.0 * np.arcsin(np.sqrt(frac))        # exact: P(wt=1) = frac
        qc.mcry(theta, ctrl_qubits, wt[0])            # multi-controlled Ry
        for q in zeros:
            qc.x(q)
    return qc, (x, y, flag, wt), weight_max



def _measured_weighted_circuit(qc, regs):
    """Attach uniform prep + flag/wt measurements to a weighted region circuit."""
    x, y, flag, wt = regs
    cf = ClassicalRegister(1, "cf")
    cw = ClassicalRegister(1, "cw")
    circ = QuantumCircuit(wt, flag, x, y, cf, cw)
    circ.h(x); circ.h(y)
    circ.compose(qc, inplace=True)
    circ.measure(flag[0], cf[0])
    circ.measure(wt[0],   cw[0])
    return circ


def _counts_to_expectation(counts, weight_max, shots):
    n_flag = n_joint = 0
    for bits, c in counts.items():
        cw_bits, cf_bits = bits.split()        # "cw cf": later-registered creg on the left
        if cf_bits == '1':
            n_flag += c
            if cw_bits == '1':
                n_joint += c
    area_fraction   = n_flag / shots
    weighted_expect = (n_joint / shots) * weight_max
    return area_fraction, weighted_expect


def quantum_centroid(m, predicate, shots, _sim=None):
    """
    Estimate the region centroid by amplitude-encoding the coordinate weights
    i and j. Builds and transpiles each weighted circuit ONCE (Fix B), then
    samples -- so repeated calls / shot sweeps don't re-transpile the heavy
    multi-controlled-Ry circuit.
    """
    sim = _sim or AerSimulator()

    # Build + transpile each weighted circuit a single time.
    qc_i, regs_i, wmax_i = weighted_region_circuit(m, predicate, lambda i, j: i)
    qc_j, regs_j, wmax_j = weighted_region_circuit(m, predicate, lambda i, j: j)

    tcirc_i = transpile(_measured_weighted_circuit(qc_i, regs_i).decompose(reps=3), sim)
    tcirc_j = transpile(_measured_weighted_circuit(qc_j, regs_j).decompose(reps=3), sim)

    counts_i = sim.run(tcirc_i, shots=shots).result().get_counts()
    counts_j = sim.run(tcirc_j, shots=shots).result().get_counts()

    area, num_i = _counts_to_expectation(counts_i, wmax_i, shots)
    _,    num_j = _counts_to_expectation(counts_j, wmax_j, shots)

    ibar = num_i / area
    jbar = num_j / area
    return (ibar, jbar), area



def build_observable_circuit(A, x, f):
    f = f / np.linalg.norm(f)

    # Get alpha from Pauli decomposition
    pauli_split = SparsePauliOp.from_operator(A)
    alpha = np.sum(np.abs(pauli_split.coeffs))
    num_system = int(np.ceil(np.log2(A.shape[0])))

    # Post-selected system state: A|x> / ||A|x>||
    Ax = A @ x
    Ax_norm = Ax / np.linalg.norm(Ax)

    # System-only circuit -- no ancilla
    qc = QuantumCircuit(num_system)

    # Step 1: Prepare A|x>/||A|x>||
    qc.append(StatePreparation(Ax_norm.astype(complex), label='A|x>'), range(num_system))

    # Step 2: U_f^dag -- rotates |f> -> |0>
    qc.append(StatePreparation(f.astype(complex), label='Uf').inverse(), range(num_system))

    # Step 3: X gates -- IQAE's good state is |1...1>, ours is |0...0>
    for i in range(num_system):
        qc.x(i)

    p_success = np.linalg.norm(Ax)**2 / alpha**2
    metadata = {
        'alpha': alpha,
        'num_system': num_system,
        'num_ancilla': 0,
        'p_success': p_success,
        'good_qubits': list(range(num_system)),
    }
    return qc, metadata

def build_grover_operator(A_circuit, good_state_qubits):
    """
    Build the Grover operator Q from circuit A.
    good_state_qubits: indices of qubits that define the good subspace
    """
    grover_op = GroverOperator(oracle=A_circuit,
                               reflection_qubits=good_state_qubits)
    return grover_op

def myQAE(A_circuit, good_state_qubits, m, nShots=10000):
    """
    Quantum Amplitude Estimation.
    A_circuit   : QuantumCircuit preparing the state
    good_state_qubits: list of qubit indices defining the good subspace
    m           : number of precision qubits
    Returns     : estimated amplitude a_tilde
    """
    n = A_circuit.num_qubits
    prec_reg = QuantumRegister(m, 'prec')
    sys_reg  = QuantumRegister(n, 'sys')
    c_reg    = ClassicalRegister(m, 'c')
    qc = QuantumCircuit(prec_reg, sys_reg, c_reg)

    # Step 1: State preparation
    qc.append(A_circuit, sys_reg)

    # Step 2: Hadamard on precision register
    for j in range(m):
        qc.h(prec_reg[j])

    # Step 3: Controlled Q^{2^j}
    Q = build_grover_operator(A_circuit, good_state_qubits)
    for j in range(m):
        power = 2**j
        Q_pow = Q.power(power).control(1)
        qc.append(Q_pow, [prec_reg[j]] + list(sys_reg))

    # Step 4: Inverse QFT on precision register
    iqft = QFT(num_qubits=m, inverse=True)
    qc.append(iqft, prec_reg)

    # Step 5: Measure precision register
    qc.measure(prec_reg, c_reg)

    # Execute and decode
    sim = AerSimulator()

    result = sim.run(qc, shots=nShots).result()
    counts = result.get_counts()

    # Decode most likely outcome
    top_bitstring = max(counts, key=counts.get)
    phi_tilde = int(top_bitstring, 2) / 2**m
    theta_tilde = np.pi * phi_tilde
    a_tilde = np.sin(theta_tilde)**2

    return a_tilde, phi_tilde, counts
"""
Chapter 14: Matrix Encoding Functions

Companion code for the matrix-encoding chapter.  Two families of block
encoding are provided:

  * Pauli expansion  -- LCU_Ax / Pauli_Block_Encoding / LCU_fTAx.
    A is expanded as sum_k c_k P_k and encoded with PREP / SELECT / UNPREP.
  * Shift decomposition -- Structured_LCU_Ax / Dirichlet_LCU_Ax and friends.
    Circulant stencils are encoded directly from the cyclic shift, at a fixed
    two-qubit ancilla register and a tight subnormalization, for every N.

Register convention, used by every builder in this file: the **ancilla
register is declared first**, so ancilla qubits are the LEAST significant bits
of the statevector and the system qubits are the most significant.  The
ancilla-|0> block is therefore obtained by striding the statevector with
2**num_ancilla (reported as metadata['ancilla_zero_stride']), not by taking a
leading slice.
"""

import numpy as np
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import StatePreparation, DiagonalGate
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.library.standard_gates import PhaseGate, XGate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

from Chapter08_QuantumGates_functions import (simulate_statevector, simulate_measurements, runCircuitOnIBMQuantum)


def LCU_Ax(A, x, mode='statevector'):
    """Implements the LCU method to compute A|x> via Prep-Select-Unprep framework.

    Convention: the ancilla register is declared FIRST in the QuantumCircuit, so
    ancilla qubits are the least significant bits of the statevector and the
    system qubits are the most significant.  Post-selection on ancilla = |0>
    therefore takes every 2**num_ancilla-th amplitude,

        A|x> / alpha  ==  Statevector(qc).data[::2**num_ancilla],

    reported as metadata['ancilla_zero_stride'].  It is NOT the leading
    2**num_system entries -- that slice mixes ancilla branches and is wrong.

    Args:
        A (np.ndarray): Hermitian operator
        x (np.ndarray): Input state vector (normalized)
        mode (str): 'statevector' or 'measurement'

    Returns:
        qc (QuantumCircuit): Quantum circuit implementing the LCU
        metadata (dict): Contains alpha, num_system, num_ancilla, coeffs, pauli_split
    """

    # Pauli decomposition
    pauli_split = SparsePauliOp.from_operator(A)
    coeffs = pauli_split.coeffs
    alpha = np.sum(np.abs(coeffs))

    L = len(coeffs)
    # ceil(log2 L) is 0 when A is a single Pauli term; keep one ancilla so that
    # PREP/UNPREP and the ancilla-|0> post-selection stay well defined.
    num_ancilla = max(int(np.ceil(np.log2(L))), 1)
    num_system = int(np.ceil(np.log2(A.shape[0])))

    # Ancilla declared first -> least significant bits in statevector
    # System declared second -> most significant bits in statevector
    qr_anc = QuantumRegister(num_ancilla, 'a')
    qr_sys = QuantumRegister(num_system, 's')
    
    if mode == 'measurement':
        cr_sys = ClassicalRegister(num_system, 'c_sys')
        cr_anc = ClassicalRegister(num_ancilla, 'c_anc')
        qc = QuantumCircuit(qr_anc, qr_sys, cr_anc, cr_sys)
    else:
        qc = QuantumCircuit(qr_anc,qr_sys) 

    # Initialize |x>_sys ⊗ |0>_anc
    qc.append(StatePreparation(x), qr_sys)

    # PREP on ancilla - encode sqrt(|c_k| / alpha)
    prep_vec = np.pad(np.sqrt(np.abs(coeffs) / alpha), (0, 2**num_ancilla - L))
    qc.append(StatePreparation(prep_vec, label='Prep'), qr_anc)

    # SELECT: ancilla controls system
    # ctrl_gate qubit order: [control_qubits..., target_qubits...]
    # => [*qr_anc, *qr_sys]: ancilla=control, system=target  ✓
    for i, (pauli, coeff) in enumerate(zip(pauli_split.paulis, coeffs)):
        phase = np.angle(coeff)
        pauli_circ = QuantumCircuit(num_system, global_phase=phase)
        pauli_circ.append(pauli.to_instruction(), range(num_system))
        ctrl_gate = pauli_circ.to_gate(
            label=f'{pauli}(φ={phase:.2f})'
        ).control(
            num_ancilla,
            ctrl_state=format(i, f'0{num_ancilla}b')
        )
        # control=ancilla, target=system
        qc.append(ctrl_gate, [*qr_anc, *qr_sys])

    # UNPREP: inverse PREP on ancilla
    qc.append(StatePreparation(prep_vec, label='Prep').inverse(), qr_anc)

    # ancilla is least significant => ancilla=|0> 
    if mode == 'measurement':
        qc = qc.decompose(reps=3)
        qc.measure(qr_sys, cr_sys)
        qc.measure(qr_anc, cr_anc)

    metadata = {
        'alpha': float(np.real(alpha)),          # real by construction (sum of moduli)
        'num_system': num_system,
        'num_ancilla': num_ancilla,
        'coeffs': coeffs,
        'pauli_split': pauli_split,
        'ancilla_zero_stride': 2**num_ancilla,
    }
    return qc, metadata


def Pauli_Block_Encoding(A, mode='statevector'):
    """Constructs a block-encoding of a Hermitian operator A using Pauli decomposition.
    
    Args:
        A (np.ndarray): Hermitian operator
        mode (str): 'statevector' or 'measurement'

    Returns:
        U_matrix (np.ndarray): Unitary matrix of the block-encoding
        metadata (dict): Contains alpha, num_system, num_ancilla, coeffs, pauli_split
    """
    x = np.zeros(A.shape[0])  # dummy state vector
    x[0] = 1.0                # |0> state
    qc, metadata = LCU_Ax(A, x, mode=mode)
    U_matrix = Operator(qc).data
    return U_matrix, metadata


def LCU_fTAx_circuit(f, A, x, add_x_gates=False):
    """
    Build the measurement-free circuit whose all-zeros amplitude is f^T A x / alpha.

    This is LCU_Ax followed by U_f^dagger on the system register, and nothing
    else -- no measurements, so the circuit is a genuine state preparation
    unitary.  That matters for two reasons:

      * amplitude estimation (Chapter 15) has to invert the state preparation
        to build the Grover operator Q = (U S_0 U^dag) S_chi; a circuit
        containing `measure` cannot be inverted;
      * U_f^dagger acts on the system register ONLY, so it commutes with any
        measurement of the ancilla register.  Measuring the ancilla first is a
        bookkeeping choice of the sampling estimator, never a requirement --
        a system-only unitary cannot entangle anything with the ancilla, and
        the ancilla-failure branch has zero overlap with |0>_anc both before
        and after U_f^dagger.

    The resulting amplitude on the all-zeros string of the FULL register is

        <0|_anc <0|_sys  U_obs  |0> = f^T A x / alpha,

    so the unconditional probability is p = |f^T A x|^2 / alpha^2 and the
    observable is recovered as |f^T A x| = alpha * sqrt(p).  Note this is the
    *unconditional* probability: it already contains the post-selection cost,
    so no extra factor of sqrt(p_success) is applied (contrast Eq. 14.16,
    which belongs to the two-stage post-select-then-sample estimator below).

    Args:
        f (np.ndarray): Observable vector (normalized internally).
        A (np.ndarray): Hermitian matrix.
        x (np.ndarray): Input vector (normalized internally).
        add_x_gates (bool): Append an X to every qubit, moving the target
            amplitude from |0...0> to |1...1>.  Set True when handing the
            circuit to Qiskit's EstimationProblem, whose good state is
            |1> on each objective qubit; the flip must cover the ancilla
            qubits as well as the system qubits, since the good state is
            all-zeros on both registers.

    Returns:
        qc (QuantumCircuit): Measurement-free circuit on
            num_ancilla + num_system qubits.
        metadata (dict): LCU_Ax metadata plus 'good_qubits' (every qubit index)
            and 'p_success' = ||Ax||^2 / alpha^2 (diagnostic; it does NOT enter
            the recovery -- see the note above).
    """
    f = np.asarray(f, dtype=complex)
    f = f / np.linalg.norm(f)
    x = np.asarray(x, dtype=complex)
    x = x / np.linalg.norm(x)

    qc, metadata = LCU_Ax(A, x, mode='statevector')
    qr_sys = qc.qregs[1]                      # ancilla first, system second

    qc.append(StatePreparation(f, label='f').inverse(), qr_sys)

    if add_x_gates:
        qc.x(range(qc.num_qubits))

    metadata['good_qubits'] = list(range(qc.num_qubits))
    metadata['p_success'] = float(
        np.linalg.norm(np.asarray(A) @ x) ** 2 / metadata['alpha'] ** 2)
    return qc, metadata


def recover_observable(p, metadata):
    """|f^T A x| = alpha * sqrt(p) from the unconditional all-zeros probability.

    Use with LCU_fTAx_circuit / amplitude estimation.  There is deliberately no
    sqrt(p_success) factor here: p is already unconditional.  Equation (14.16),
    |f^T A x| = sqrt(p_0) * alpha * sqrt(p_success), applies to the two-stage
    estimator of LCU_fTAx, where p_0 is measured *conditional* on the ancilla
    having been post-selected to |0>.
    """
    return metadata['alpha'] * np.sqrt(np.clip(np.real(p), 0.0, 1.0))


def LCU_fTAx(f, A, x, shots=10000, noise_model=None):
    """
    Estimate |f^T A x| by sampling the LCU observable circuit.

    Two-stage estimator: post-select the shots whose ancilla read |0...0>, then
    among those count the fraction whose system read |0...0> after U_f^dagger.
    That fraction is the conditional p_0 = |f^T A x|^2 / ||Ax||^2, and

        |f^T A x| = sqrt(p_0) * ||Ax||,     ||Ax|| = alpha * sqrt(p_success),

    which is Equation (14.16).  Both registers are measured together at the end
    of the circuit; the post-selection is done classically in the counting loop
    below.  (An earlier version measured the ancilla mid-circuit, before
    U_f^dagger.  That was unnecessary -- U_f^dagger touches only the system
    register, so it commutes with an ancilla measurement -- and it made the
    circuit non-invertible, which blocked its reuse for amplitude estimation.
    See LCU_fTAx_circuit.)

    Args:
        f (np.ndarray): Observable vector (will be normalized)
        A (np.ndarray): Matrix
        x (np.ndarray): Input vector
        shots (int): Number of measurements
        noise_model: Optional noise model passed to the simulator.

    Returns:
        inner_product (float): Estimate of |f^T * A * x|
        qc (QuantumCircuit): Full measurement circuit
        metadata (dict): Circuit metadata including success_prob
    """

    # Normalize f
    f = f / np.linalg.norm(f)

    # Step 1: LCU circuit + U_f^dagger, still measurement-free
    qc, metadata = LCU_fTAx_circuit(f, A, x)

    num_system = metadata['num_system']
    num_ancilla = metadata['num_ancilla']

    qr_anc = qc.qregs[0]  # ancilla: declared first in QuantumCircuit(qr_anc, qr_sys)
    qr_sys = qc.qregs[1]  # system:  declared second

    # Step 2: Add classical registers
    cr_anc = ClassicalRegister(num_ancilla, 'c_anc')
    cr_sys = ClassicalRegister(num_system, 'c_sys')
    qc.add_register(cr_anc)
    qc.add_register(cr_sys)

    # Step 3: Measure both registers at the end.  Post-selection on
    # ancilla = |0...0> happens classically, in the counting loop below.
    qc.measure(qr_anc, cr_anc)
    qc.measure(qr_sys, cr_sys)

    # Step 4: Run circuit
    counts = simulate_measurements(qc, shots=shots, noise_model=noise_model)

    # Step 5: Post-process
    ancilla_zero = '0' * num_ancilla
    system_zero  = '0' * num_system
    alpha = metadata['alpha']

    count_proj = 0
    total_postselected = 0

    for outcome, count in counts.items():
        # Qiskit bit string order: 'c_sys c_anc' (last register is leftmost)
        parts = outcome.split(' ')
        sys_bits = parts[0]   # c_sys is added second -> leftmost in string
        anc_bits = parts[1]   # c_anc is added first  -> rightmost in string

        if anc_bits == ancilla_zero:
            total_postselected += count
            if sys_bits == system_zero:
                count_proj += count

    success_prob = total_postselected / shots
    metadata['success_prob'] = success_prob

    if total_postselected > 0:
        prob_f = count_proj / total_postselected
        norm_Ax = alpha * np.sqrt(success_prob)
        inner_product = np.sqrt(prob_f) * norm_Ax
        return inner_product, qc, metadata
    else:
        return 0.0, qc, metadata

# =====================================================================
#  Shift-decomposition encoding of periodic / circulant matrices
#  Book §14.9 -- Examples: the periodic and Dirichlet 1D Laplacians
# =====================================================================


def periodic_laplacian(N, dtype=float):
    """Periodic (circulant) 1D Laplacian on N nodes with wrap-around neighbours.

        A_per = 2 I - S_c - S_c^dagger

    i.e. 2 on the diagonal, -1 on both off-diagonals, and -1 in the two
    corners that close the ring.

    Args:
        N (int): Number of nodes. Must be a power of two and >= 4
                 (for N = 2 the two off-diagonals collapse onto each other).

    Returns:
        A (np.ndarray): N x N periodic Laplacian.
    """
    return circulant_tridiagonal(N, a=2.0, b=-1.0, dtype=dtype)


def circulant_tridiagonal(N, a=2.0, b=-1.0, dtype=complex):
    """Generic periodic tridiagonal (circulant) stencil

        A = a I + b S_c + conj(b) S_c^dagger,

    where S_c is the cyclic shift.  A is Hermitian by construction.  The
    periodic Laplacian is the case (a, b) = (2, -1).

    The eigenvalues are known in closed form,
        lambda_k = a + 2 |b| cos(2 pi k / N + arg b),  k = 0, ..., N-1,
    which is what makes the tightness of the subnormalization checkable.

    Args:
        N (int): Matrix size, power of two, >= 4.
        a (complex): Diagonal entry (must be real for A to be Hermitian).
        b (complex): Subdiagonal entry; the superdiagonal is conj(b).

    Returns:
        A (np.ndarray): N x N circulant Hermitian matrix.
    """
    if N < 4 or (N & (N - 1)) != 0:
        raise ValueError("N must be a power of two and at least 4.")
    if abs(np.imag(a)) > 1e-12:
        raise ValueError("The diagonal coefficient a must be real for A to be Hermitian.")

    S = np.zeros((N, N), dtype=complex)
    for j in range(N):
        S[(j + 1) % N, j] = 1.0          # S|j> = |(j+1) mod N>

    A = a * np.eye(N, dtype=complex) + b * S + np.conj(b) * S.conj().T
    A = np.real_if_close(A)
    if dtype is not complex and not np.iscomplexobj(A):
        A = A.astype(dtype)          # downcast only when genuinely real
    return A


def cyclic_shift_circuit(m, power=+1):
    """Circuit for the cyclic shift S_c |j> = |(j+1) mod N> on m qubits, N = 2**m.

    S_c is the *modular incrementer*: it adds one modulo N, so it is unitary and
    can be used directly as an LCU term -- no Pauli expansion required.  It is
    built here as the textbook ripple cascade: a descending sequence of
    multi-controlled X gates followed by a single X on the least significant
    qubit,

        S_c = X_0 . CX_{0->1} . CCX_{01->2} ... ,

    which for m = 2 is exactly the two-gate circuit S_c = X_0 CX_{0->1}.

    This is the primitive itself.  The SELECT stage of the block encodings below
    applies the same cascade *controlled* on the ancilla register, gate by gate;
    see _append_select_shift().  An alternative synthesis, not used here, writes
    the incrementer in Fourier form as QFT^dag . diag(exp(2 pi i k / N)) . QFT and
    controls only the phases.

    Args:
        m (int): Number of system qubits, m >= 2. N = 2**m.
        power (int): +1 for S_c, -1 for S_c^dagger (the modular decrementer).

    Returns:
        qc (QuantumCircuit): m-qubit circuit implementing S_c^{power}.
    """
    if m < 2:
        raise ValueError("m must be at least 2 (N >= 4).")
    if power not in (+1, -1):
        raise ValueError("power must be +1 (S_c) or -1 (S_c^dagger).")

    name = 'S_c' if power == +1 else 'S_c†'
    qc = QuantumCircuit(m, name=name)
    for k in range(m - 1, 0, -1):
        qc.mcx(list(range(k)), k)
    qc.x(0)
    if power == -1:
        qc = qc.inverse()
        qc.name = name
    return qc


def _append_select_shift(qc, qr_anc, qr_sys, coeffs, labels=None):
    """SELECT stage for the three-term shift decomposition: apply S_c on ancilla
    |01>, S_c^dagger on ancilla |10>, and nothing on |00> (the identity term) or
    on the unused |11>.

    The shift is the modular incrementer of cyclic_shift_circuit(), applied
    *controlled* on the ancilla register.  The control is distributed over the
    individual gates of the cascade rather than wrapped around the cascade as a
    whole: controlled-(g_k ... g_1) = (controlled-g_k) ... (controlled-g_1), and
    writing it out this way lets the transpiler see the structure.  (Wrapping
    the cascade in a single .control() call instead is logically identical but
    transpiles roughly an order of magnitude worse in Qiskit.)

    Each gate of the cascade therefore becomes a multi-controlled X carrying the
    two ancilla qubits as extra controls.  This is the cost driver of the whole
    encoding: with an ancilla-assisted synthesis of those multi-controlled X
    gates, each shift costs O(m) Toffoli gates and the encoding is O(log N)
    overall, which is the figure quoted in the text.  Qiskit's default
    ancilla-free synthesis is more expensive than that bound; encoding_resources()
    reports what a given Qiskit build actually costs.

    Args:
        qc (QuantumCircuit): Circuit being built (modified in place).
        qr_anc, qr_sys (QuantumRegister): Ancilla (2 qubits) and system registers.
        coeffs (np.ndarray): The three LCU coefficients (a, b, conj(b)).
        labels (list): Unused; retained so callers may name the terms.
    """
    m = len(qr_sys)
    na = len(qr_anc)
    # the incrementer, gate by gate: MCX cascade from the top, then X on qubit 0
    steps = [(list(range(k)), k) for k in range(m - 1, 0, -1)]

    for i, power in [(1, +1), (2, -1)]:            # |01> -> S_c, |10> -> S_c^dag
        if abs(coeffs[i]) == 0.0:
            continue
        if power == -1:                            # S_c^dag = (X_0 . cascade)^dag
            qc.mcx(list(qr_anc), qr_sys[0], ctrl_state=i)
        for ctl, tgt in (steps if power == +1 else list(reversed(steps))):
            # controls are [a_0, ..., a_{na-1}, s_ctl...]; the system controls are
            # all positive, the ancilla pattern is the integer i
            ctrl_state = i | ((2 ** len(ctl) - 1) << na)
            qc.mcx([*qr_anc, *[qr_sys[c] for c in ctl]], qr_sys[tgt],
                   ctrl_state=ctrl_state)
        if power == +1:
            qc.mcx(list(qr_anc), qr_sys[0], ctrl_state=i)


def _sign_pattern_as_z_string(signs):
    """Return the list of qubit indices carrying a Z, if the sign vector is a
    product of Z gates; otherwise return None.

    A length-2**n real sign vector s (entries +-1, indexed by the ancilla
    integer k) is realized by Z gates on a subset V of the ancilla qubits iff
        s[k] = (-1)**(number of bits of k inside V) = prod_{t in V} (-1)**k_t.
    For the periodic Laplacian the required pattern is (+, -, -, +), which is
    exactly Z on *both* ancilla qubits -- the "two Z gates" of the text.
    """
    signs = np.asarray(signs)
    n = int(np.log2(len(signs)))
    if not np.all(np.isin(signs, [1.0, -1.0])) or signs[0] != 1.0:
        return None
    qubits = [t for t in range(n) if signs[2 ** t] < 0]
    for k in range(len(signs)):
        parity = sum((k >> t) & 1 for t in qubits) % 2
        if signs[k] != (-1) ** parity:
            return None
    return qubits


def Structured_LCU_Ax(N, x=None, a=2.0, b=-1.0, mode='statevector'):
    """Shift-decomposition LCU block-encoding of the circulant stencil

        A = a I + b S_c + conj(b) S_c^dagger        (a = 2, b = -1 -> periodic Laplacian)

    and, optionally, the application of A to a state |x>.

    Instead of expanding A in the Pauli basis (whose term count grows with N),
    the structure is read off directly: a constant diagonal plus two constant
    off-diagonals that wrap around.  The three terms {I, S_c, S_c^dagger} are
    *already unitary*, so LCU applies verbatim:

        PREP    : two ancilla qubits load sqrt(|c_k| / alpha), k = 0, 1, 2
                  (the fourth ancilla state is unused and carries amplitude 0)
        SIGNS   : the coefficient phases are injected on the ancilla -- for
                  real coefficients with pattern (+, -, -) this is literally
                  Z on each of the two ancilla qubits, i.e. Z (x) Z
        SELECT  : controlled-S_c on ancilla |01>, controlled-S_c^dagger on |10>,
                  the incrementer cascade carrying the ancilla as extra controls
                  (the |00> term is the identity and costs nothing)
        UNPREP  : PREP^dagger

    The ancilla register is *fixed at two qubits for every N*, and the
    subnormalization is
        alpha = |a| + 2|b|,
    which for (a, b) = (2, -1) is alpha = 4, matching ||A_per||_2 = 4 exactly
    for every even N.

    Convention (identical to LCU_Ax): the ancilla register is declared first,
    so ancilla qubits are the least significant bits of the statevector and
    the ancilla-|0> block is obtained by striding with 2**num_ancilla.

    Args:
        N (int): Matrix size, power of two, >= 4.
        x (np.ndarray or None): Normalized system state to act on.  If None,
            no state preparation is appended -- use this to read off the
            block-encoding unitary itself.
        a (float): Diagonal coefficient.
        b (complex): Subdiagonal coefficient (superdiagonal is conj(b)).
        mode (str): 'statevector' or 'measurement'.

    Returns:
        qc (QuantumCircuit): The PREP-SELECT-UNPREP circuit.
        metadata (dict): alpha, num_system, num_ancilla, coeffs, terms,
            ancilla_zero_stride, A (the dense matrix, for checking),
            sign_qubits (which ancilla qubits carry a Z).
    """
    if N < 4 or (N & (N - 1)) != 0:
        raise ValueError("N must be a power of two and at least 4.")

    num_system = int(np.log2(N))
    num_ancilla = 2                      # three terms -> two ancilla qubits, for every N

    coeffs = np.array([a, b, np.conj(b)], dtype=complex)
    terms = ['I', 'S_c', 'S_c^dag']       # ASCII, for printing
    glabels = ['I', 'S_c', 'S_c†']            # for the drawn circuit
    L = len(coeffs)
    alpha = float(np.sum(np.abs(coeffs)))
    if alpha == 0.0:
        raise ValueError("All coefficients are zero.")

    qr_anc = QuantumRegister(num_ancilla, 'a')
    qr_sys = QuantumRegister(num_system, 's')

    if mode == 'measurement':
        cr_anc = ClassicalRegister(num_ancilla, 'c_anc')
        cr_sys = ClassicalRegister(num_system, 'c_sys')
        qc = QuantumCircuit(qr_anc, qr_sys, cr_anc, cr_sys)
    else:
        qc = QuantumCircuit(qr_anc, qr_sys)

    # ---- optional state preparation |x>_sys -------------------------------
    if x is not None:
        x = np.asarray(x, dtype=complex)
        x = x / np.linalg.norm(x)
        qc.append(StatePreparation(x, label='x'), qr_sys)

    # ---- PREP: load the coefficient magnitudes ----------------------------
    prep_vec = np.pad(np.sqrt(np.abs(coeffs) / alpha), (0, 2 ** num_ancilla - L))
    qc.append(StatePreparation(prep_vec, label='PREP'), qr_anc)

    # ---- SIGNS: inject the coefficient phases on the ancilla --------------
    phases = np.angle(coeffs)
    phase_vec = np.concatenate([np.exp(1j * phases), np.ones(2 ** num_ancilla - L)])
    sign_qubits = None
    if np.allclose(phase_vec.imag, 0.0, atol=1e-12):
        sign_qubits = _sign_pattern_as_z_string(np.real(phase_vec))
    if sign_qubits is not None:
        for t in sign_qubits:                 # (2, -1, -1) -> Z on both ancilla qubits
            qc.z(qr_anc[t])
    else:
        qc.append(DiagonalGate(list(phase_vec)), qr_anc)

    # ---- SELECT: ancilla |01> -> S_c,  ancilla |10> -> S_c^dagger ---------
    _append_select_shift(qc, qr_anc, qr_sys, coeffs, labels=glabels)

    # ---- UNPREP -----------------------------------------------------------
    qc.append(StatePreparation(prep_vec, label='PREP').inverse(), qr_anc)

    if mode == 'measurement':
        qc.measure(qr_anc, cr_anc)
        qc.measure(qr_sys, cr_sys)

    metadata = {
        'alpha': alpha,
        'num_system': num_system,
        'num_ancilla': num_ancilla,
        'coeffs': coeffs,
        'terms': terms,
        'ancilla_zero_stride': 2 ** num_ancilla,
        'A': circulant_tridiagonal(N, a=a, b=b),
        'sign_qubits': sign_qubits,
    }
    return qc, metadata


def Structured_Block_Encoding(N, a=2.0, b=-1.0):
    """Block-encoding unitary of the circulant stencil built from the cyclic shift.

    Returns the full unitary U of the PREP-SIGNS-SELECT-UNPREP circuit; the
    ancilla-|0> block satisfies  (<0|_anc (x) I) U (|0>_anc (x) I) = A / alpha.
    With the ancilla as least significant bits that block is U[::2**k, ::2**k]
    with k = num_ancilla.

    Args:
        N (int): Matrix size, power of two, >= 4.
        a, b (complex): Stencil coefficients.

    Returns:
        U_matrix (np.ndarray): Unitary of the block-encoding.
        metadata (dict): Same fields as Structured_LCU_Ax.
    """
    qc, metadata = Structured_LCU_Ax(N, x=None, a=a, b=b, mode='statevector')
    U_matrix = Operator(qc).data
    return U_matrix, metadata


def extract_encoded_block(U, num_ancilla, alpha):
    """Recover the encoded matrix alpha * <0|_anc U |0>_anc from a block-encoding.

    Assumes the ancilla register occupies the least significant qubits
    (the convention used by LCU_Ax and Structured_LCU_Ax).

    Args:
        U (np.ndarray): Block-encoding unitary.
        num_ancilla (int): Number of ancilla qubits.
        alpha (float): Subnormalization.

    Returns:
        A_rec (np.ndarray): The recovered matrix.
    """
    stride = 2 ** num_ancilla
    return U[::stride, ::stride] * alpha


def verify_block_encoding(U, A, alpha, num_ancilla, tol=1e-8, verbose=True):
    """Check that U really block-encodes A with subnormalization alpha.

    Reports the reconstruction error, whether U is unitary, and how tight the
    subnormalization is (alpha / ||A||_2; 1.0 means tight).

    Args:
        U (np.ndarray): Block-encoding unitary.
        A (np.ndarray): Target matrix.
        alpha (float): Subnormalization.
        num_ancilla (int): Number of ancilla qubits.
        tol (float): Tolerance for the pass/fail verdict.

    Returns:
        report (dict): error, is_unitary, spectral_norm, tightness, passed.
    """
    A_rec = extract_encoded_block(U, num_ancilla, alpha)
    err = np.max(np.abs(A_rec - A))
    is_unitary = np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-8)
    nrm = np.linalg.norm(A, 2)
    report = {
        'error': err,
        'is_unitary': is_unitary,
        'spectral_norm': nrm,
        'alpha': alpha,
        'tightness': alpha / nrm,
        'passed': bool(err < tol and is_unitary),
    }
    if verbose:
        print(f"  block-encoding error   : {err:.3e}")
        print(f"  U unitary              : {is_unitary}")
        print(f"  alpha / ||A||_2        : {alpha:.4f} / {nrm:.4f} = {report['tightness']:.4f}")
        print(f"  verdict                : {'PASS' if report['passed'] else 'FAIL'}")
    return report


def encoding_resources(qc, basis_gates=('u', 'cx'), optimization_level=1):
    """Transpiled resource count for a circuit: depth, total gates, CX count.

    Args:
        qc (QuantumCircuit): Circuit to analyse.
        basis_gates (tuple): Target basis for transpilation.
        optimization_level (int): Qiskit transpiler optimization level.

    Returns:
        res (dict): qubits, depth, gates, cx.
    """
    from qiskit import transpile
    qct = transpile(qc, basis_gates=list(basis_gates),
                    optimization_level=optimization_level)
    ops = qct.count_ops()
    return {
        'qubits': qc.num_qubits,
        'depth': qct.depth(),
        'gates': int(sum(ops.values())),
        'cx': int(ops.get('cx', 0)),
    }


def compare_periodic_encodings(N_list=(4, 8, 16, 32), transpile_resources=True,
                               verbose=True):
    """Side-by-side cost of the two block-encodings of the periodic Laplacian.

    Route 1 (standard)   : Pauli expansion A = sum_k c_k P_k, encoded with LCU_Ax.
                           The term count grows like 3N/4 and the subnormalization
                           like alpha = log2(N) + 2, while ||A||_2 stays at 4 --
                           so the encoding gets progressively looser.
    Route 2 (shift decomp): A = 2 I - S_c - S_c^dagger, encoded with the cyclic
                           shift.  Three terms and two ancilla qubits for every N,
                           at the tight alpha = 4.

    Args:
        N_list (iterable): Matrix sizes, powers of two >= 4.
        transpile_resources (bool): Include transpiled depth/gate counts.

    Returns:
        rows (list of dict): One record per N.
    """
    rows = []
    for N in N_list:
        A = periodic_laplacian(N)
        nrm = np.linalg.norm(A, 2)

        # ---- Route 1: Pauli expansion ------------------------------------
        sp = SparsePauliOp.from_operator(A).simplify()
        n_pauli = len(sp.coeffs)
        alpha_p = float(np.sum(np.abs(sp.coeffs)))
        anc_p = int(np.ceil(np.log2(n_pauli)))

        row = {
            'N': N, 'm': int(np.log2(N)), 'norm': nrm,
            'pauli_terms': n_pauli, 'pauli_ancilla': anc_p,
            'pauli_alpha': alpha_p, 'pauli_tightness': alpha_p / nrm,
            'struct_terms': 3, 'struct_ancilla': 2,
            'struct_alpha': 4.0, 'struct_tightness': 4.0 / nrm,
        }

        if transpile_resources:
            x = np.zeros(N); x[0] = 1.0
            qc_p, _ = LCU_Ax(A, x, mode='statevector')
            qc_s, _ = Structured_LCU_Ax(N, x=None)
            rp = encoding_resources(qc_p)
            rs = encoding_resources(qc_s)
            row.update({'pauli_qubits': rp['qubits'], 'pauli_depth': rp['depth'],
                        'pauli_gates': rp['gates'], 'pauli_cx': rp['cx'],
                        'struct_qubits': rs['qubits'], 'struct_depth': rs['depth'],
                        'struct_gates': rs['gates'], 'struct_cx': rs['cx']})
        rows.append(row)

    if verbose:
        head = (f"{'N':>5} | {'terms':>6} {'anc':>4} {'alpha':>6} {'depth':>7} {'CX':>7}"
                f" | {'terms':>6} {'anc':>4} {'alpha':>6} {'depth':>7} {'CX':>7}")
        print(f"{'':>5} | {'Pauli expansion':^34} | {'Shift decomposition':^34}")
        print(head)
        print('-' * len(head))
        for r in rows:
            print(f"{r['N']:>5} | {r['pauli_terms']:>6} {r['pauli_ancilla']:>4} "
                  f"{r['pauli_alpha']:>6.2f} {r.get('pauli_depth', -1):>7} "
                  f"{r.get('pauli_cx', -1):>7} | {r['struct_terms']:>6} "
                  f"{r['struct_ancilla']:>4} {r['struct_alpha']:>6.2f} "
                  f"{r.get('struct_depth', -1):>7} {r.get('struct_cx', -1):>7}")
    return rows


def plot_encoding_scaling(rows, figsize=(11, 3.6)):
    """Three-panel scaling figure for the periodic-Laplacian encodings:
    LCU term count, subnormalization alpha against the exact norm, and
    transpiled two-qubit gate count -- all versus N.

    N is always a power of two, so the horizontal axis is ticked at the actual
    problem sizes rather than at matplotlib's decade minor ticks, which collide
    and are unreadable over this range.

    Args:
        rows (list of dict): Output of compare_periodic_encodings().
        figsize (tuple): Figure size.

    Returns:
        fig, axes: Matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter

    Ns = [r['N'] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ax = axes[0]
    ax.plot(Ns, [r['pauli_terms'] for r in rows], 'o-', label='Pauli expansion')
    ax.plot(Ns, [r['struct_terms'] for r in rows], 's--', label='shift decomposition')
    ax.set_yscale('log'); ax.set_ylabel('number of LCU terms')
    yt = sorted({r['pauli_terms'] for r in rows} | {r['struct_terms'] for r in rows})
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(round(v))}'))
    ax.set_title('Terms in the decomposition'); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(Ns, [r['norm'] for r in rows], '-', color='0.6', lw=4,
            label=r'$\|A\|_2$ (exact)')
    ax.plot(Ns, [r['pauli_alpha'] for r in rows], 'o-', label=r'$\alpha$ (Pauli)')
    ax.plot(Ns, [r['struct_alpha'] for r in rows], 's--', label=r'$\alpha$ (shift decomp.)')
    ax.set_ylabel('subnormalization')
    ax.set_title('Tightness of the encoding'); ax.legend(fontsize=8)

    ax = axes[2]
    if 'pauli_cx' in rows[0]:
        ax.plot(Ns, [r['pauli_cx'] for r in rows], 'o-', label='Pauli expansion')
        ax.plot(Ns, [r['struct_cx'] for r in rows], 's--', label='shift decomposition')
    ax.set_yscale('log'); ax.set_ylabel('CX gates after transpilation')
    ax.set_title('Two-qubit gate cost'); ax.legend(fontsize=8)

    # one x-axis treatment for all three panels: log base 2, ticked at the data
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.xaxis.set_major_locator(FixedLocator(Ns))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{int(round(v))}'))
        ax.set_xlabel('N')
        ax.grid(True, which='major', alpha=0.25, lw=0.5)

    fig.tight_layout()
    return fig, axes


def dirichlet_laplacian(N, a=2.0, b=-1.0, dtype=float):
    """Open-boundary (Dirichlet) tridiagonal stencil on N interior nodes:
    the circulant matrix with the two wrap-around corner entries removed.

    Args:
        N (int): Matrix size, power of two, >= 4.
        a (float): Diagonal coefficient.
        b (complex): Subdiagonal coefficient (superdiagonal is conj(b)).

    Returns:
        A (np.ndarray): N x N tridiagonal matrix.
    """
    A = circulant_tridiagonal(N, a=a, b=b)
    A[0, N - 1] = 0.0                 # wrap-around from S_c
    A[N - 1, 0] = 0.0                 # wrap-around from S_c^dagger
    A = np.real_if_close(A)
    if dtype is not complex and not np.iscomplexobj(A):
        A = A.astype(dtype)          # downcast only when genuinely real
    return A


def Dirichlet_LCU_Ax(N, x=None, a=2.0, b=-1.0, mode='statevector'):
    """Structured encoding of the *Dirichlet* (open-boundary) stencil, obtained
    from the periodic circuit by flagging the wrap-around contributions into a
    failure subspace.
    
    The correction below is *not* the construction of Kharazi et al., who
    compose each shift with a reflection about the boundary node (their
    Eq. 49, a five-term LCU).  We instead flag the wrap-around contributions
    into a failure subspace.  Both reach (alpha = 4, 3 ancillas); the flag
    route keeps three LCU terms and spends the third ancilla on the flag,
    the reflection route spends it on the larger LCU register.  The flag is
    used here because it reuses the periodic SELECT verbatim.

    The decomposition is not re-derived: it is still the three-term
        A = a I + b S_c + conj(b) S_c^dagger
    at the same subnormalization alpha = |a| + 2|b|.  What changes is a single
    extra *flag* ancilla, set between SELECT and UNPREP whenever the shift
    actually wrapped around:

        ancilla |01> (S_c applied)        -> the wrapped component is the one
                                             that lands on |0...0>
        ancilla |10> (S_c^dagger applied) -> the wrapped component is the one
                                             that lands on |1...1>

    SELECT itself is the controlled incrementer cascade of _append_select_shift(),
    shared verbatim with the periodic encoder.

    Each check is one multi-controlled X onto the flag with m + 2 controls, and
    the flag is left set (not uncomputed) so that post-selecting on
    flag = 0 discards exactly the two corner entries.  The encoded block is then
        alpha * <0|_anc <0|_flag U |0>_anc |0>_flag = A_dirichlet.

    The correction is circuit-level, not decomposition-level: no LCU term is
    added, the ancilla count rises by exactly one for every N, and the
    m-dependence sits in the multi-controlled flag gate.

    Note that alpha = 4 is no longer tight for the Dirichlet stencil:
    ||A||_2 = 2 - 2 cos(N pi / (N+1)) < 4, approaching 4 only as N -> infinity.

    Args:
        N (int): Matrix size, power of two, >= 4.
        x (np.ndarray or None): Normalized system state, or None for the bare
            block-encoding.
        a (float): Diagonal coefficient.
        b (complex): Subdiagonal coefficient.
        mode (str): 'statevector' or 'measurement'.

    Returns:
        qc (QuantumCircuit): The circuit, with registers (a, flag, s).
        metadata (dict): alpha, num_system, num_ancilla (LCU + flag),
            num_lcu_ancilla, ancilla_zero_stride, A, terms, coeffs.

    
    """
    if N < 4 or (N & (N - 1)) != 0:
        raise ValueError("N must be a power of two and at least 4.")

    num_system = int(np.log2(N))
    num_lcu_anc = 2
    num_ancilla = num_lcu_anc + 1                 # + one flag qubit

    coeffs = np.array([a, b, np.conj(b)], dtype=complex)
    terms = ['I', 'S_c', 'S_c^dag']
    glabels = ['I', 'S_c', 'S_c†']
    alpha = float(np.sum(np.abs(coeffs)))

    qr_anc = QuantumRegister(num_lcu_anc, 'a')
    qr_flg = QuantumRegister(1, 'f')
    qr_sys = QuantumRegister(num_system, 's')

    if mode == 'measurement':
        cr_anc = ClassicalRegister(num_lcu_anc, 'c_anc')
        cr_flg = ClassicalRegister(1, 'c_flg')
        cr_sys = ClassicalRegister(num_system, 'c_sys')
        qc = QuantumCircuit(qr_anc, qr_flg, qr_sys, cr_anc, cr_flg, cr_sys)
    else:
        qc = QuantumCircuit(qr_anc, qr_flg, qr_sys)

    if x is not None:
        x = np.asarray(x, dtype=complex)
        x = x / np.linalg.norm(x)
        qc.append(StatePreparation(x, label='x'), qr_sys)

    # ---- PREP + signs (identical to the periodic case) --------------------
    prep_vec = np.pad(np.sqrt(np.abs(coeffs) / alpha), (0, 2 ** num_lcu_anc - len(coeffs)))
    qc.append(StatePreparation(prep_vec, label='PREP'), qr_anc)

    phases = np.angle(coeffs)
    phase_vec = np.concatenate([np.exp(1j * phases), np.ones(2 ** num_lcu_anc - len(coeffs))])
    sign_qubits = None
    if np.allclose(phase_vec.imag, 0.0, atol=1e-12):
        sign_qubits = _sign_pattern_as_z_string(np.real(phase_vec))
    if sign_qubits is not None:
        for t in sign_qubits:
            qc.z(qr_anc[t])
    else:
        qc.append(DiagonalGate(list(phase_vec)), qr_anc)

    # ---- SELECT (identical to the periodic case) --------------------------
    _append_select_shift(qc, qr_anc, qr_sys, coeffs, labels=glabels)

    # ---- FLAG the two wrap-around contributions ---------------------------
    # controls are ordered [a0, a1, s0, ..., s_{m-1}]; ctrl_state is read with
    # bit i of the integer belonging to control i.
    ctrls = [*qr_anc, *qr_sys]
    if abs(coeffs[1]) > 0.0:                      # S_c wrapped -> landed on |0...0>
        state_inc = 1                             # a = 01, system all zeros
        qc.append(XGate().control(len(ctrls), ctrl_state=state_inc), [*ctrls, qr_flg[0]])
    if abs(coeffs[2]) > 0.0:                      # S_c† wrapped -> landed on |1...1>
        state_dec = 2 | ((2 ** num_system - 1) << 2)   # a = 10, system all ones
        qc.append(XGate().control(len(ctrls), ctrl_state=state_dec), [*ctrls, qr_flg[0]])

    # ---- UNPREP -----------------------------------------------------------
    qc.append(StatePreparation(prep_vec, label='PREP').inverse(), qr_anc)

    if mode == 'measurement':
        qc.measure(qr_anc, cr_anc)
        qc.measure(qr_flg, cr_flg)
        qc.measure(qr_sys, cr_sys)

    metadata = {
        'alpha': alpha,
        'num_system': num_system,
        'num_ancilla': num_ancilla,
        'num_lcu_ancilla': num_lcu_anc,
        'coeffs': coeffs,
        'terms': terms,
        'ancilla_zero_stride': 2 ** num_ancilla,
        'A': dirichlet_laplacian(N, a=a, b=b),
        'sign_qubits': sign_qubits,
    }
    return qc, metadata


def Dirichlet_Block_Encoding(N, a=2.0, b=-1.0):
    """Block-encoding unitary of the Dirichlet stencil (periodic circuit + flag).

    Args:
        N (int): Matrix size, power of two, >= 4.
        a, b (complex): Stencil coefficients.

    Returns:
        U_matrix (np.ndarray): Unitary of the block-encoding.
        metadata (dict): Same fields as Dirichlet_LCU_Ax.
    """
    qc, metadata = Dirichlet_LCU_Ax(N, x=None, a=a, b=b, mode='statevector')
    return Operator(qc).data, metadata
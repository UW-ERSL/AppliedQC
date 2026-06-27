import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Arc
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit.circuit.library import GroverOperator
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import SparsePauliOp
from qiskit import  transpile
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.primitives import StatevectorSampler

from Chapter08_QuantumGates_functions import (simulate_statevector, simulate_measurements, runCircuitOnIBMQuantum,
                                              findActualHardwareRequirements, plot_measurement_results)

from Chapter14_MatrixEncoding_functions import LCU_Ax

"""
Iterative Quantum Amplitude Estimation (IQAE) -- classical statistics demo.

This reproduces the worked-by-hand example and draws the theta confidence
bracket for each round.  No quantum circuit / Grover operator is built: the
ONLY role Grover plays is to make a measurement succeed with probability
sin^2((2k+1)*theta).  We sample that directly with a binomial draw, which is
all that is needed to illustrate the scheduling and interval-squeezing logic.

Set fixed_counts=[23, 36, 51] to reproduce the exact figure in the text;
set fixed_counts=None and pick any max_rounds to let the reader run their own.
"""

import numpy as np
import matplotlib.pyplot as plt

HALF_PI = np.pi / 2.0


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def confidence_interval(n_success, N, z=1.96):
    """95% Wald interval on the success probability (z = 1.96)."""
    p_hat = n_success / N
    se = np.sqrt(p_hat * (1.0 - p_hat) / N)
    lo = max(0.0, p_hat - z * se)
    hi = min(1.0, p_hat + z * se)
    return p_hat, lo, hi


# ----------------------------------------------------------------------
# Scheduling: largest power k whose magnified bracket stays on one
# monotone arc of sin^2 (turning points sit at multiples of pi/2).
# ----------------------------------------------------------------------
def largest_valid_k(theta_lo, theta_hi, k_cap=None):
    width = theta_hi - theta_lo
    if width <= 0:
        k_max = 10**6
    else:
        k_max = max(int(((HALF_PI / width) - 1.0) / 2.0), 0)
    if k_cap is not None:
        k_max = min(k_max, k_cap)
    for k in range(k_max, -1, -1):
        f = 2 * k + 1
        m_lo = int(np.floor(f * theta_lo / HALF_PI + 1e-12))
        m_hi = int(np.floor(f * theta_hi / HALF_PI - 1e-12))
        if m_lo == m_hi:                       # both ends on the same arc
            return k, m_lo
    return 0, int(np.floor(theta_lo / HALF_PI))


# ----------------------------------------------------------------------
# Inversion: probability interval -> theta interval on arc m.
# ----------------------------------------------------------------------
def invert(p_lo, p_hi, k, m):
    f = 2 * k + 1
    a_lo = np.arcsin(np.sqrt(np.clip(p_lo, 0.0, 1.0)))
    a_hi = np.arcsin(np.sqrt(np.clip(p_hi, 0.0, 1.0)))
    left = m * HALF_PI
    if m % 2 == 0:                              # rising arc
        phi_lo, phi_hi = left + a_lo, left + a_hi
    else:                                       # falling arc: endpoints swap
        phi_lo, phi_hi = left + (HALF_PI - a_hi), left + (HALF_PI - a_lo)
    return phi_lo / f, phi_hi / f


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
def iqae(theta_true, N=100, z=1.96, eps_target=0.0, max_rounds=3,
         seed=0, fixed_counts=None, k_cap=None):
    rng = np.random.default_rng(seed)
    theta_lo, theta_hi = 0.0, HALF_PI
    rounds = []
    for r in range(max_rounds):
        k, m = largest_valid_k(theta_lo, theta_hi, k_cap)
        f = 2 * k + 1
        p_true = np.sin(f * theta_true) ** 2
        if fixed_counts is not None and r < len(fixed_counts):
            n_succ = int(fixed_counts[r])
        else:
            n_succ = int(rng.binomial(N, p_true))
        p_hat, p_lo, p_hi = confidence_interval(n_succ, N, z)
        t_lo, t_hi = invert(p_lo, p_hi, k, m)
        new_lo, new_hi = max(theta_lo, t_lo), min(theta_hi, t_hi)
        if new_lo < new_hi:                     # normal case: intersect
            theta_lo, theta_hi = new_lo, new_hi
        else:                                   # rare statistical miss
            theta_lo, theta_hi = t_lo, t_hi
            print(f"  [round {r+1}: confidence miss, interval reset]")
        width_p = np.sin(theta_hi) ** 2 - np.sin(theta_lo) ** 2
        rounds.append(dict(r=r + 1, k=k, f=f, n=n_succ, N=N, p_hat=p_hat,
                           p_true=p_true,
                           arc="rising" if m % 2 == 0 else "falling",
                           theta_lo=theta_lo, theta_hi=theta_hi,
                           width_p=width_p))
        if eps_target > 0.0 and width_p < eps_target:
            break
    return rounds


def print_table(rounds):
    head = f"{'Rnd':>3} {'k':>3} {'2k+1':>5} {'succ/N':>8} {'p_hat':>6} " \
           f"{'p_true':>7} {'arc':>8} {'theta bracket':>22} {'width_p':>8}"
    print(head)
    print("-" * len(head))
    for x in rounds:
        br = f"[{x['theta_lo']:.3f}, {x['theta_hi']:.3f}]"
        print(f"{x['r']:>3} {x['k']:>3} {x['f']:>5} "
              f"{str(x['n'])+'/'+str(x['N']):>8} {x['p_hat']:>6.2f} "
              f"{x['p_true']:>7.3f} {x['arc']:>8} {br:>22} {x['width_p']:>8.4f}")


def plot_rounds_linear(rounds, theta_true):
    fig, ax = plt.subplots(figsize=(7.0, 0.7 * len(rounds) + 1.4))
    cap = 0.18
    for x in rounds:
        y = x["r"]
        ax.plot([x["theta_lo"], x["theta_hi"]], [y, y],
                lw=5, color="#2c6fbb", solid_capstyle="butt", alpha=0.85)
        ax.plot([x["theta_lo"]] * 2, [y - cap, y + cap], color="#2c6fbb", lw=1.5)
        ax.plot([x["theta_hi"]] * 2, [y - cap, y + cap], color="#2c6fbb", lw=1.5)
        ax.text(x["theta_hi"] + 0.004, y,
                f"$k={x['k']}$, width$_p$={x['width_p']:.3f}",
                va="center", ha="left", fontsize=8.5)
    ax.axvline(theta_true, color="#c0392b", ls="--", lw=1.4,
               label=fr"true $\theta = {theta_true}$")
    r1 = rounds[0]
    pad_l, pad_r = 0.05, 0.14
    ax.set_xlim(r1["theta_lo"] - pad_l, r1["theta_hi"] + pad_r)
    ax.set_yticks([x["r"] for x in rounds])
    ax.set_yticklabels([f"Round {x['r']}" for x in rounds])
    ax.invert_yaxis()
    ax.set_xlabel(r"amplitude angle  $\theta$")
    ax.set_title("IQAE confidence bracket on $\\theta$, by round")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    ax.grid(axis="x", ls=":", alpha=0.4)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return

def plot_rounds_wedge(rounds, theta_true):
    """Wedge figure from iqae() output.
    `rounds` is the list of dicts returned by iqae(); each has keys
    r, k, theta_lo, theta_hi, width_p.  Top row = true scale, bottom = zoomed."""
    R = 1.0
    BLUE, RED = '#2c6fbb', '#c0392b'
    FS_TITLE, FS_KET = 28, 27

    ncols = len(rounds)
    fig, axes = plt.subplots(2, ncols, figsize=(5.7 * ncols, 12))
    if ncols == 1:                       # keep axes 2D when only one round
        axes = axes.reshape(2, 1)

    def title_for(x):
        return f"Round {x['r']}  ($k={x['k']}$)\nwidth$_p$ = {x['width_p']:.3f}"

    # ---------- ROW 0: WITHOUT zoom -- full plane, true scale ----------
    for ax, x in zip(axes[0], rounds):
        lo, hi = x['theta_lo'], x['theta_hi']
        ax.plot([0, R], [0, 0], color='0.55', lw=2.0)
        ax.plot([0, 0], [0, R], color='0.55', lw=2.0)
        ax.add_patch(Arc((0, 0), 2*R, 2*R, theta1=0, theta2=90, color='0.8', lw=1.6))
        ax.add_patch(Wedge((0, 0), R, np.degrees(lo), np.degrees(hi),
                           facecolor=BLUE, alpha=0.35, edgecolor='none'))
        for e in (lo, hi):
            ax.plot([0, R*np.cos(e)], [0, R*np.sin(e)], color=BLUE, lw=2.4)
        ax.plot([0, R*np.cos(theta_true)], [0, R*np.sin(theta_true)],
                color=RED, ls='--', lw=2.4)
        ax.text(1.03, -0.03, r"$|s^\perp\rangle$", fontsize=FS_KET, va='top')
        ax.text(-0.03, 1.06, r"$|s\rangle$", fontsize=FS_KET, ha='right')
        ax.set_xlim(-0.10, 1.22); ax.set_ylim(-0.10, 1.20)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title_for(x), fontsize=FS_TITLE)

    # ---------- ROW 1: WITH zoom -- each panel magnifies ----------
    for ax, x in zip(axes[1], rounds):
        lo, hi = x['theta_lo'], x['theta_hi']
        cx = np.degrees(theta_true)
        half = max(np.degrees(hi - lo) * 1.8, 2.5)
        a0, a1 = cx - half, cx + half
        ax.plot([0, R*np.cos(np.radians(a0))], [0, R*np.sin(np.radians(a0))], color='0.85', lw=1.6)
        ax.plot([0, R*np.cos(np.radians(a1))], [0, R*np.sin(np.radians(a1))], color='0.85', lw=1.6)
        ax.add_patch(Arc((0, 0), 2*R, 2*R, theta1=a0, theta2=a1, color='0.6', lw=1.6))
        ax.add_patch(Wedge((0, 0), R, np.degrees(lo), np.degrees(hi),
                           facecolor=BLUE, alpha=0.30, edgecolor='none'))
        for e in (lo, hi):
            ax.plot([0, R*np.cos(e)], [0, R*np.sin(e)], color=BLUE, lw=2.6)
        ax.plot([0, R*np.cos(theta_true)], [0, R*np.sin(theta_true)],
                color=RED, ls='--', lw=2.6)
        ax.set_xlim(np.cos(np.radians(a1))*R - 0.02, np.cos(np.radians(a0))*R + 0.02)
        ax.set_ylim(np.sin(np.radians(a0))*R - 0.02, np.sin(np.radians(a1))*R + 0.02)
        ax.set_aspect('equal'); ax.axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(hspace=0.55)
   
    return fig



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
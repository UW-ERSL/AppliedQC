"""
Chapter 15: Quantum Amplitude Estimation (QAE) - Functions

Companion code for the Quantum Amplitude Estimation chapter. It provides the
Iterative Amplitude Estimation (IQAE) statistics demo -- Wald confidence
intervals, the power (2k+1) scheduling that keeps the magnified bracket on one
monotone arc of sin^2, and the inversion from a probability interval back to a
θ interval -- along with the table/wedge plotting helpers used for the book's
figures and a circuit builder that recasts an observable estimate fᵀA|x⟩ into
the IQAE "good state" framework.
"""

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
    """
    Pick the largest Grover power k whose magnified bracket stays on one arc.

    Amplifying by the odd factor f = 2k+1 stretches the angle bracket
    [theta_lo, theta_hi] to [f*theta_lo, f*theta_hi]. The inversion in
    :func:`invert` is only unambiguous while both endpoints fall on the same
    monotone arc of sin^2, whose turning points sit at integer multiples of π/2.
    This searches k downward from its theoretical maximum and returns the first
    (largest) k for which both scaled endpoints land in the same arc.

    Parameters
    ----------
    theta_lo : float
        Lower end of the current θ bracket, in radians (0 ≤ theta_lo ≤ π/2).
    theta_hi : float
        Upper end of the current θ bracket, in radians.
    k_cap : int, optional
        Upper limit on k (e.g. a hardware depth budget). If None, no extra cap
        beyond the arc-width constraint is applied.

    Returns
    -------
    k : int
        The chosen Grover power (≥ 0).
    m : int
        Index of the arc of sin^2 both scaled endpoints occupy, i.e.
        ``floor(f*theta / (π/2))``. Even m is a rising arc, odd m a falling arc.
    """
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
    """
    Map a success-probability interval back to a θ interval on arc m.

    Inverts the relation p = sin^2(f·θ), with f = 2k+1, on the single monotone
    arc identified by ``m``. Each probability bound is turned into an angle via
    arcsin(√p); on a rising arc (even m) the order of the bounds is preserved,
    while on a falling arc (odd m) the endpoints swap. The recovered angles are
    de-magnified by dividing by f to give the estimate for θ itself.

    Parameters
    ----------
    p_lo : float
        Lower confidence bound on the success probability (clipped to [0, 1]).
    p_hi : float
        Upper confidence bound on the success probability (clipped to [0, 1]).
    k : int
        Grover power, so the magnification factor is f = 2k+1.
    m : int
        Arc index from :func:`largest_valid_k`; its parity selects rising vs.
        falling inversion.

    Returns
    -------
    theta_lo : float
        Lower end of the inverted θ interval, in radians.
    theta_hi : float
        Upper end of the inverted θ interval, in radians.
    """
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
    """
    Run the Iterative Amplitude Estimation loop and record each round.

    Simulates IQAE without building any quantum circuit: the sole role of the
    Grover operator -- boosting the good-state probability to sin^2(f·θ) with
    f = 2k+1 -- is reproduced with a binomial draw. Each round schedules the
    largest valid power k, samples (or reads) a success count, forms a Wald
    confidence interval, inverts it to a θ bracket, and intersects it with the
    running bracket, squeezing the estimate of the amplitude angle θ.

    Parameters
    ----------
    theta_true : float
        The true amplitude angle θ (radians) that the binomial sampling targets.
    N : int, optional
        Number of shots per round used in the binomial draw. Default 100.
    z : float, optional
        Gaussian z-score for the Wald confidence interval. Default 1.96 (95%).
    eps_target : float, optional
        Early-stop tolerance on the probability-domain bracket width ``width_p``;
        the loop breaks once it falls below this. Default 0.0 (no early stop).
    max_rounds : int, optional
        Maximum number of IQAE rounds to run. Default 3.
    seed : int, optional
        Seed for the NumPy random generator. Default 0.
    fixed_counts : sequence of int, optional
        Predetermined success counts per round (used to reproduce the exact
        worked example in the text). Rounds beyond its length fall back to
        random sampling. If None, all counts are sampled.
    k_cap : int, optional
        Cap on the Grover power k, forwarded to :func:`largest_valid_k`.

    Returns
    -------
    list of dict
        One dict per round with keys ``r`` (round number), ``k``, ``f`` (=2k+1),
        ``n`` (success count), ``N``, ``p_hat``, ``p_true``, ``arc``
        ("rising"/"falling"), ``theta_lo``, ``theta_hi``, and ``width_p`` (the
        probability-domain bracket width).
    """
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
    """
    Print the per-round IQAE summary as an aligned text table.

    Parameters
    ----------
    rounds : list of dict
        The round records returned by :func:`iqae`. Each is expected to carry the
        keys ``r``, ``k``, ``f``, ``n``, ``N``, ``p_hat``, ``p_true``, ``arc``,
        ``theta_lo``, ``theta_hi`` and ``width_p``.

    Returns
    -------
    None
        The table is written to standard output.
    """
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
    """
    Draw the IQAE confidence brackets on a shared linear θ axis.

    Renders one horizontal error-bar per round showing its [theta_lo, theta_hi]
    bracket (annotated with k and the probability-domain width), with the true
    angle marked as a dashed vertical reference. This is the linear counterpart
    to :func:`plot_rounds_wedge`.

    Parameters
    ----------
    rounds : list of dict
        Round records from :func:`iqae`; each uses ``r``, ``k``, ``theta_lo``,
        ``theta_hi`` and ``width_p``.
    theta_true : float
        True amplitude angle θ (radians), drawn as the dashed reference line.

    Returns
    -------
    None
        The figure is created via matplotlib as a side effect and not returned.
    """
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
        """
        Build the two-line subplot title for one round record ``x``.

        Parameters
        ----------
        x : dict
            A single round record from :func:`iqae`, using its ``r``, ``k`` and
            ``width_p`` fields.

        Returns
        -------
        str
            A LaTeX-formatted title of the form ``Round r ($k=..$)`` with the
            probability-domain width on the second line.
        """
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
    """
    Build the system-only circuit that casts fᵀA|x⟩ into IQAE's good state.

    Prepares the post-selected system state A|x⟩/‖A|x⟩‖, rotates the observable
    direction |f⟩ onto |0…0⟩ with U_f†, and applies X on every qubit so the
    target amplitude sits on |1…1⟩ -- the "good state" IQAE expects. No ancilla
    is used; the returned metadata reports the theoretical success probability
    ‖A|x⟩‖² / α² (with α the L1 norm of the Pauli coefficients of A).

    Parameters
    ----------
    A : numpy.ndarray, shape (2**n, 2**n)
        Operator acting on the system register; its Pauli decomposition sets the
        normalization constant α.
    x : numpy.ndarray, shape (2**n,)
        Input state vector.
    f : numpy.ndarray, shape (2**n,)
        Observable direction vector; normalized internally before use.

    Returns
    -------
    qc : qiskit.QuantumCircuit
        System-only circuit preparing the IQAE good state.
    metadata : dict
        Keys ``alpha``, ``num_system``, ``num_ancilla`` (0), ``p_success`` and
        ``good_qubits`` (the list of system-qubit indices).
    """
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

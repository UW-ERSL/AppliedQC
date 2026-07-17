"""
Variational Quantum Linear Solver (VQLS)
========================================
Solves the quantum linear system problem --- find |u> = |x/||x||> such that
A x = b --- with a hybrid quantum-classical loop (Book Chapter 21):

  1. a parametric ansatz U(theta) generates trial states |u(theta)> = U(theta)|0>,
  2. A is expanded as a linear combination of Pauli (unitary) operators,
  3. a cost function scores each trial (lowest when A|u(theta)> is parallel to b),
  4. a classical optimizer (COBYLA) finds the optimal theta*.

Global cost
-----------
This module implements the *global* cost

    C_G(theta) = 1 - |<b|A|u(theta)>|^2 / <u(theta)|A^dag A|u(theta)>,

evaluated exactly from the statevector. The observation that makes this an
ordinary (ancilla-free) expectation value is that the numerator is *quadratic*
in |u>:

    |<b|A|u>|^2 = <u| A^dag |b><b| A |u> = <u| M |u>,   M = A^dag |b><b| A,

and M is Hermitian, so a single estimator call measures it. This is the correct
and efficient way to obtain the cost on a *simulator*.

IMPORTANT (simulator vs. hardware). Forming M requires the projector |b><b|,
i.e. knowing b's amplitudes classically. That is fine for the statevector
demonstrations here, but is exactly what a real device cannot do at scale. On
hardware, <b|A|u> and <u|A^dag A|u> are instead assembled term by term from the
Pauli expansion A = sum_k a_k P_k, via the ancilla-based Hadamard tests of
Book Section 21.4 --- one pair of tests (real, imaginary) per g_j and per
h_{jk}, giving the O(L^2) circuit count discussed in the text. The statevector
cost below is a faithful stand-in for that estimate, not a replacement for the
hardware routine. The barren-plateau-resistant local cost C_L is discussed in
the text by reference only and is not implemented here.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import real_amplitudes as RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize


class VQLS:
    """
    Variational Quantum Linear Solver with a global cost (exact statevector by
    default, or shot-based when ``nShots`` is given).

    Parameters
    ----------
    A : (N, N) array_like
        System matrix, N a power of two. Expanded internally as a SparsePauliOp.
    b : (N,) array_like
        Right-hand side. Normalized internally; only its direction matters.
    reps : int, optional
        Ansatz repetition layers for ``real_amplitudes``. Default 6.
    entanglement : str, optional
        Entanglement pattern for the ansatz. Default ``'full'``.
    nShots : int or None, optional
        If ``None`` (default), the cost is evaluated exactly from the
        statevector. If an integer, the two expectation values are instead
        sampled with a shot-based estimator, so the cost carries the finite-shot
        noise that a real device would exhibit (used to reproduce the
        shot-count examples in the text).

    Attributes
    ----------
    ansatz : QuantumCircuit
        The parametric ansatz U(theta).
    num_qubits : int
        log2(N).
    """

    def __init__(self, A, b, reps=6, entanglement="full", nShots=None):
        self.A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).flatten()
        self.b = b / np.linalg.norm(b)
        self.nShots = nShots

        self.num_qubits = int(np.log2(self.A.shape[0]))
        self.ansatz = RealAmplitudes(self.num_qubits, reps=reps,
                                     entanglement=entanglement)
        if nShots is None:
            self._estimator = StatevectorEstimator()
        else:
            # Shot-based estimator: samples the same operators M and A^dag A,
            # so only the estimator changes, not the cost formula.
            from qiskit_aer.primitives import EstimatorV2 as AerEstimator
            self._estimator = AerEstimator()

        P = SparsePauliOp.from_operator(self.A)                 # A as a Pauli sum
        b_proj = SparsePauliOp.from_operator(np.outer(self.b, self.b.conj()))
        self._AdA = P.adjoint().compose(P).simplify()          # A^dag A
        # M = A^dag |b><b| A  (Hermitian); <u|M|u> = |<b|A|u>|^2
        self._M = P.adjoint().compose(b_proj).compose(P).simplify()

    def cost(self, theta):
        """
        Global VQLS cost C_G(theta).

        Computes C_G = 1 - <u|M|u> / <u|A^dag A|u>, with M = A^dag |b><b| A;
        the numerator equals |<b|A|u>|^2, so the cost vanishes exactly when
        A|u> is parallel to b. Evaluated exactly from the statevector when
        ``nShots`` is None, or sampled with finite shots otherwise.

        Parameters
        ----------
        theta : array_like
            Real ansatz parameters, one per ``self.ansatz.num_parameters``.

        Returns
        -------
        float
            Cost in [0, 1]; 0 means A|u(theta)> is parallel to b.
        """
        bound = self.ansatz.assign_parameters(theta)
        if self.nShots is None:
            num = float(np.real(self._estimator.run([(bound, self._M)])
                                .result()[0].data.evs))
            den = float(np.real(self._estimator.run([(bound, self._AdA)])
                                .result()[0].data.evs))
        else:
            precision = 1.0 / np.sqrt(self.nShots)
            num = float(np.real(self._estimator.run(
                [(bound, self._M)], precision=precision).result()[0].data.evs))
            den = float(np.real(self._estimator.run(
                [(bound, self._AdA)], precision=precision).result()[0].data.evs))
        return 1.0 - num / den

    def solve(self, initial_theta=None, method="COBYLA", maxiter=3000, seed=None):
        """
        Minimize the global cost to recover the normalized solution direction.

        Parameters
        ----------
        initial_theta : array_like, optional
            Starting parameters; random if None.
        method : str, optional
            SciPy optimizer name. Default ``'COBYLA'``.
        maxiter : int, optional
            Maximum optimizer iterations. Default 3000.
        seed : int, optional
            Seed for the random start.

        Returns
        -------
        u : numpy.ndarray
            The recovered normalized state |u(theta*)>.
        result : scipy.optimize.OptimizeResult
            The optimizer result (``result.fun`` final cost, ``result.x`` angles).
        """
        if initial_theta is None:
            rng = np.random.default_rng(seed)
            initial_theta = rng.random(self.ansatz.num_parameters)
        result = minimize(self.cost, initial_theta, method=method,
                          options={"maxiter": maxiter})
        u = Statevector(self.ansatz.assign_parameters(result.x)).data
        return u, result

    def classical_solution(self):
        """Normalized classical solution x/||x|| of A x = b, for verification."""
        x = np.linalg.solve(self.A, self.b)
        return x / np.linalg.norm(x)

    @staticmethod
    def fidelity(u, v):
        """State fidelity |<u|v>|^2 between two normalized vectors."""
        return float(np.abs(np.vdot(u, v)) ** 2)
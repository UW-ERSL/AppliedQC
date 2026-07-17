"""
Variational Quantum Linear Solver (VQLS)
========================================
Solves the quantum linear system problem --- find |u> = |x/||x||> such that
A x = b --- with a hybrid quantum-classical loop (Book Chapter 21):

  1. a parametric ansatz U(theta) generates trial states |u(theta)> = U(theta)|0>,
  2. A is expanded as a linear combination of Pauli (unitary) operators,
  3. a cost function scores each trial (lowest when A|u(theta)> is parallel to b),
  4. a classical optimizer searches for the best theta*.

Global cost
-----------
This module implements the *global* cost

    C_G(theta) = 1 - |<b|A|u(theta)>|^2 / <u(theta)|A^dag A|u(theta)>,

evaluated exactly from the statevector. The observation that makes this an
ordinary (ancilla-free) expectation value is that the numerator is *quadratic*
in |u>:

    |<b|A|u>|^2 = <u| A^dag |b><b| A |u> = <u| M |u>,   M = A^dag |b><b| A,

and M is Hermitian, so a single estimator call measures it. This is the correct
way to obtain the overlap on a *simulator*.

IMPORTANT (simulator vs. hardware). Forming the operator M requires the
projector |b><b|, i.e. knowing b's amplitudes classically. That is fine for the
statevector demonstrations here, but it is exactly what a real quantum device
cannot do at scale. On hardware, <b|A|u> must instead be estimated with the
ancilla-based Hadamard test described in the text (Section 21.4); the
statevector cost below is a faithful stand-in for that estimate, not a
replacement for the hardware routine.

The barren-plateau-resistant *local* cost C_L is discussed in the text by
reference only and is not implemented here.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import real_amplitudes as RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize


class VQLS:
    """
    Variational Quantum Linear Solver with an exact-statevector global cost.

    Parameters
    ----------
    A : (N, N) array_like
        System matrix, with N a power of two. Expanded internally as a
        SparsePauliOp.
    b : (N,) array_like
        Right-hand side. Normalized internally; only its direction matters.
    reps : int, optional
        Number of ansatz repetition layers for ``real_amplitudes``. Deeper
        ansatze are more expressive but have more parameters. Default 6.
    entanglement : str, optional
        Entanglement pattern passed to ``real_amplitudes`` (e.g. ``'full'``,
        ``'linear'``). Default ``'full'``.

    Attributes
    ----------
    ansatz : QuantumCircuit
        The parametric ansatz U(theta).
    num_qubits : int
        log2(N).
    """

    def __init__(self, A, b, reps=6, entanglement="full"):
        self.A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float).flatten()
        self.b = b / np.linalg.norm(b)

        self.num_qubits = int(np.log2(self.A.shape[0]))
        self.ansatz = RealAmplitudes(self.num_qubits, reps=reps,
                                     entanglement=entanglement)
        self._estimator = StatevectorEstimator()

        # Operator forms used by the cost function.
        P = SparsePauliOp.from_operator(self.A)                 # A as a Pauli sum
        b_proj = SparsePauliOp.from_operator(np.outer(self.b, self.b.conj()))
        self._AdA = P.adjoint().compose(P).simplify()          # A^dag A
        # M = A^dag |b><b| A  (Hermitian); <u|M|u> = |<b|A|u>|^2
        self._M = P.adjoint().compose(b_proj).compose(P).simplify()

    def cost(self, theta):
        """
        Global VQLS cost C_G(theta), evaluated exactly from the statevector.

        Computes

            C_G = 1 - <u|M|u> / <u|A^dag A|u>,   M = A^dag |b><b| A,

        where |u> = U(theta)|0>. The numerator equals |<b|A|u>|^2, so the cost
        vanishes exactly when A|u> is parallel to b.

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
        num = float(np.real(self._estimator.run([(bound, self._M)])
                            .result()[0].data.evs))
        den = float(np.real(self._estimator.run([(bound, self._AdA)])
                            .result()[0].data.evs))
        return 1.0 - num / den

    def solve(self, initial_theta=None, method="COBYLA", maxiter=3000, seed=None):
        """
        Minimize the global cost to recover the normalized solution direction.

        Parameters
        ----------
        initial_theta : array_like, optional
            Starting parameters. If ``None``, drawn uniformly at random.
        method : str, optional
            SciPy optimizer name. Default ``'COBYLA'`` (derivative-free).
        maxiter : int, optional
            Maximum optimizer iterations. Default 3000.
        seed : int, optional
            Seed for the random initial point (used only if
            ``initial_theta`` is None).

        Returns
        -------
        u : numpy.ndarray
            The recovered normalized state |u(theta*)> (the quantum solution).
        result : scipy.optimize.OptimizeResult
            The full optimizer result (``result.fun`` is the final cost,
            ``result.x`` the optimal parameters).
        """
        if initial_theta is None:
            rng = np.random.default_rng(seed)
            initial_theta = rng.random(self.ansatz.num_parameters)
        result = minimize(self.cost, initial_theta, method=method,
                          options={"maxiter": maxiter})
        u = Statevector(self.ansatz.assign_parameters(result.x)).data
        return u, result

    def classical_solution(self):
        """
        Normalized classical solution of A x = b, for verification.

        Returns
        -------
        numpy.ndarray
            ``x/||x||`` where ``A x = b``.
        """
        x = np.linalg.solve(self.A, self.b)
        return x / np.linalg.norm(x)

    @staticmethod
    def fidelity(u, v):
        """
        State fidelity |<u|v>|^2 between two normalized vectors.

        Parameters
        ----------
        u, v : array_like
            Normalized state vectors.

        Returns
        -------
        float
            |<u|v>|^2 in [0, 1]; 1 indicates identical directions.
        """
        return float(np.abs(np.vdot(u, v)) ** 2)
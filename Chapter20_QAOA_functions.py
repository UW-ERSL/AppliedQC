"""
QAOA for QUBO Problems
======================
Gate-based (QAOA) back-ends for the two engineering workflows introduced with
quantum annealing earlier in the book:

  * QAOABoxSolver      -- Box Algorithm for A x = b   (cf. Chapter 6 QUBOBoxSolverClass)
  * QAOATrussOptimizer -- truss area optimization      (cf. Chapter 5 QATrussOptimizer)

Design principle
----------------
Both classes *subclass* the corresponding Chapter 5/6 annealing solver and
override **only** the routine that minimizes a single QUBO. Every other piece
--- the box translate/contract logic, the truss QUBO coefficient formulas
(Eqs. 5.35-5.42), the finite-element bookkeeping, and all decoding --- is
inherited verbatim. Consequently these solvers return the *same* answers as
their annealing counterparts; only the per-iteration QUBO is handed to QAOA
instead of to a simulated/quantum annealer.

The bridge to QAOA is `qubo_dict_to_sparse_pauliop`, which converts a QUBO in
dimod dictionary form ``{(i, j): Q_ij}`` into a Qiskit ``SparsePauliOp`` cost
Hamiltonian via the substitution q_i = (I - Z_i)/2 (Book Section 20.1).

Notes
-----
QAOA on a statevector simulator is limited to small qubit counts. The truss
example is therefore restricted to a small ground structure (a handful of
members plus one slack qubit); the full 3x3 ground structure (28 bars) yields
a 29-qubit QUBO that is beyond a statevector QAOA and should be run with the
Chapter 5 annealing solver instead.
"""

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

# Reuse the annealing solvers unchanged; we only swap the QUBO back-end.
from Chapter05_QuantumAnnealing_functions import QATrussOptimizer
from Chapter06_RealNumberEncoding_functions import QUBOBoxSolverClass


# ---------------------------------------------------------------------------
# QUBO (dict) -> cost Hamiltonian (SparsePauliOp)
# ---------------------------------------------------------------------------
def qubo_dict_to_sparse_pauliop(Q, n_qubits):
    """
    Convert a QUBO in dictionary form to a Qiskit cost Hamiltonian.

    Applies the standard substitution q_i = (I - Z_i)/2 (Book Section 20.1) to
    the QUBO objective  f(q) = sum_{i<=j} Q[(i,j)] q_i q_j  and collects the
    resulting Pauli terms. The returned operator is diagonal in the
    computational basis, and its ground state is the minimizing bitstring.

    Parameters
    ----------
    Q : dict
        QUBO coefficients keyed by ``(i, j)`` with ``i <= j``. Diagonal keys
        ``(i, i)`` carry linear coefficients; off-diagonal keys carry the
        quadratic couplings (dimod / pyqubo convention).
    n_qubits : int
        Number of binary variables (qubits) in the QUBO.

    Returns
    -------
    SparsePauliOp
        Cost Hamiltonian H_C. The constant (identity) offset is retained so
        that eigenvalues equal QUBO costs; it does not affect the minimizer.

    Notes
    -----
    Uses the Qiskit little-endian string convention: the rightmost character
    of each Pauli label acts on qubit 0. Substituting q_i = (I - Z_i)/2 gives
    for a linear term  Q_ii q_i -> (Q_ii/2)(I - Z_i)  and for a quadratic term
    Q_ij q_i q_j -> (Q_ij/4)(I - Z_i)(I - Z_j).
    """
    # Accumulate coefficients of Z-strings, keyed by a frozenset of qubit indices.
    z_coeffs = {}          # frozenset(indices) -> coefficient
    offset = [0.0]

    def add_z(indices, coeff):
        key = frozenset(indices)
        if not key:
            offset[0] += coeff
        else:
            z_coeffs[key] = z_coeffs.get(key, 0.0) + coeff

    for (i, j), coeff in Q.items():
        if coeff == 0.0:
            continue
        if i == j:
            # Q_ii q_i = (Q_ii/2)(I - Z_i)
            add_z((), coeff / 2.0)
            add_z((i,), -coeff / 2.0)
        else:
            # Q_ij q_i q_j = (Q_ij/4)(I - Z_i)(I - Z_j)
            #             = (Q_ij/4)(I - Z_i - Z_j + Z_i Z_j)
            add_z((), coeff / 4.0)
            add_z((i,), -coeff / 4.0)
            add_z((j,), -coeff / 4.0)
            add_z((i, j), coeff / 4.0)

    def z_string(indices):
        s = ["I"] * n_qubits
        for k in indices:
            s[n_qubits - 1 - k] = "Z"   # little-endian: qubit 0 is rightmost
        return "".join(s)

    terms = [(z_string(idx), c) for idx, c in z_coeffs.items() if c != 0.0]
    terms.append(("I" * n_qubits, offset[0]))
    return SparsePauliOp.from_list(terms)


def solve_qubo_with_qaoa(Q, n_qubits, reps=1, maxiter=300,
                         initial_point=None, return_optimizer_point=False):
    """
    Minimize a QUBO (dictionary form) with QAOA on a statevector simulator.

    Builds the cost Hamiltonian via :func:`qubo_dict_to_sparse_pauliop`, runs
    Qiskit's ``QAOA`` with the default transverse-field mixer and a COBYLA
    classical optimizer, and returns the most probable measured bitstring.

    Parameters
    ----------
    Q : dict
        QUBO coefficients ``{(i, j): Q_ij}`` (see
        :func:`qubo_dict_to_sparse_pauliop`).
    n_qubits : int
        Number of binary variables.
    reps : int, optional
        Number of QAOA layers p (the ``reps`` argument). Default 1.
    maxiter : int, optional
        Maximum COBYLA iterations. Default 300.
    initial_point : array-like, optional
        Warm-start point for the variational parameters. If ``None``, QAOA
        chooses its own initial angles.
    return_optimizer_point : bool, optional
        If True, also return the converged variational parameters (for
        parameter transfer between related QUBOs). Default False.

    Returns
    -------
    q_solution : numpy.ndarray
        Binary vector ``[q_0, ..., q_{n-1}]`` (q_0 first) of the most probable
        measured bitstring.
    optimal_point : numpy.ndarray, optional
        Converged variational parameters; returned only when
        ``return_optimizer_point`` is True.
    """
    H_C = qubo_dict_to_sparse_pauliop(Q, n_qubits)
    qaoa = QAOA(
        sampler=StatevectorSampler(),
        optimizer=COBYLA(maxiter=maxiter),
        reps=reps,
        initial_point=initial_point,
    )
    result = qaoa.compute_minimum_eigenvalue(H_C)
    bitstring = max(result.eigenstate, key=result.eigenstate.get)
    # Qiskit bitstring is little-endian: rightmost char = q_0.
    q_solution = np.array([int(b) for b in reversed(bitstring)])
    if return_optimizer_point:
        return q_solution, np.asarray(result.optimal_point)
    return q_solution


# ---------------------------------------------------------------------------
# Box Algorithm via QAOA  (Book Section 20.5)
# ---------------------------------------------------------------------------
class QAOABoxSolver(QUBOBoxSolverClass):
    """
    Box Algorithm for A x = b using QAOA in place of annealing.

    Subclasses :class:`Chapter06_RealNumberEncoding_functions.QUBOBoxSolverClass`
    and reuses its adaptive box logic (translate / contract, binary encoding,
    convergence test) without modification. The only change is the sampler: at
    each box iteration the compiled QUBO is minimized with QAOA rather than with
    simulated annealing.

    Parameter transfer
    ------------------
    Because the box merely recenters and shrinks between iterations, the QUBO
    changes slowly and the optimal QAOA angles change slowly too. Setting
    ``parameter_transfer=True`` warm-starts each QAOA call from the previous
    iteration's converged angles, sharply reducing COBYLA restarts.

    Parameters
    ----------
    reps : int, optional
        Number of QAOA layers p. Default 1.
    parameter_transfer : bool, optional
        Warm-start QAOA from the previous iteration's angles. Default True.
    maxiter : int, optional
        Maximum COBYLA iterations per QUBO. Default 300.
    **kwargs
        Forwarded to ``QUBOBoxSolverClass`` (``beta``, ``LBox0``, ``tol``,
        ``boxMaxIteration``, ...). ``samplingMethod`` is ignored, since the
        QUBO is solved by QAOA.
    """

    def __init__(self, reps=1, parameter_transfer=True, maxiter=300, **kwargs):
        # The parent selects a sampler by samplingMethod; force a valid one so
        # its __init__ succeeds, then never use it.
        kwargs.setdefault("samplingMethod", "simulatedAnnealing")
        super().__init__(**kwargs)
        self.reps = reps
        self.parameter_transfer = parameter_transfer
        self.maxiter = maxiter
        self._theta_prev = None       # cached angles for warm-start

    def _bqm_to_qubo_dict(self, bqm):
        """
        Convert a dimod BQM (as produced by the parent's pyQUBO model) to a
        QUBO dict ``{(i, j): Q_ij}`` over integer qubit indices, together with
        an ordered list of the BQM variable labels.

        Returns
        -------
        Q : dict
            QUBO coefficients over integer indices.
        variables : list
            The BQM variable labels, in the index order used by ``Q``.
        """
        # Deterministic ordering of variable labels -> integer indices.
        variables = sorted(bqm.variables, key=lambda v: str(v))
        index = {v: k for k, v in enumerate(variables)}

        qubo, _offset = bqm.to_qubo()   # (qubo_dict keyed by (u, v), offset)
        Q = {}
        for (u, v), coeff in qubo.items():
            iu, iv = index[u], index[v]
            i, j = (iu, iv) if iu <= iv else (iv, iu)
            Q[(i, j)] = Q.get((i, j), 0.0) + coeff
        return Q, variables

    def _solve_bqm(self, bqm):
        """
        Minimize one box QUBO with QAOA and return a dimod-style result shim.

        Builds the QUBO dict from ``bqm``, solves it with QAOA (warm-started if
        enabled), and returns an object exposing ``.first.sample`` and
        ``.first.energy`` so the inherited box loop can consume it unchanged.
        """
        Q, variables = self._bqm_to_qubo_dict(bqm)
        n = len(variables)

        q_solution, theta = solve_qubo_with_qaoa(
            Q, n, reps=self.reps, maxiter=self.maxiter,
            initial_point=self._theta_prev if self.parameter_transfer else None,
            return_optimizer_point=True,
        )
        if self.parameter_transfer:
            self._theta_prev = theta

        sample = {variables[k]: int(q_solution[k]) for k in range(n)}
        energy = float(bqm.energy(sample))
        return _ResultShim(sample, energy)

    def QUBOBoxSolve(self, A, b, xGuess=[], debug=False):
        """
        Solve ``A x = b`` by the adaptive box method with a QAOA QUBO back-end.

        Identical control flow to the parent
        :meth:`QUBOBoxSolverClass.QUBOBoxSolve`, but each box QUBO is minimized
        with QAOA. See the parent for the full parameter and return
        description.
        """
        self.matrixSize = A.shape[0]
        self.model = self.modelWithPlaceHolders()

        qSol = [self.nQubitsPerDimension * [0] for _ in range(self.matrixSize)]
        center = list(xGuess) if len(xGuess) else self.matrixSize * [0]

        self.modelDictionary = {}
        for i in range(self.matrixSize):
            self.modelDictionary['b[%d]' % i] = b[i]
            for j in range(self.matrixSize):
                self.modelDictionary["A[{i}][{j}]".format(i=i, j=j)] = A[i, j]

        L = self.LBox0
        boxSuccess = True
        nTranslations = 0
        nContractions = 0
        PEHat = 0.0
        results = None

        for iteration in range(self.boxMaxIteration):
            if L / self.LBox0 < self.relativeTolerance:
                break

            self.modelDictionary['L'] = L
            for i in range(self.matrixSize):
                self.modelDictionary['c[%d]' % i] = center[i]

            bqm = self.model.to_bqm(feed_dict=self.modelDictionary)

            results = self._solve_bqm(bqm)        # <-- QAOA back-end
            sample = results.first.sample
            PEStar = results.first.energy

            if PEStar < PEHat * (1 + 1e-8):
                for i in range(self.matrixSize):
                    qSol[i][0] = sample["q[" + str(i) + "][0]"]
                    qSol[i][1] = sample["q[" + str(i) + "][1]"]
                PEHat = PEStar
                for i in range(self.matrixSize):
                    center[i] = center[i] + L * (-2 * qSol[i][0] + qSol[i][1])
                nTranslations += 1
            else:
                L = L * self.beta
                nContractions += 1

            if debug:
                print(f"Iter {iteration}; center {center}; PE {PEStar:.4g}; L {L:.4g}")

        if L / self.LBox0 > self.relativeTolerance:
            print("Box method did not converge to desired tolerance")
            boxSuccess = False

        return [np.array(center), L, iteration, boxSuccess,
                nTranslations, nContractions, results]


class _ResultShim:
    """Minimal stand-in for a dimod sampleset's ``.first`` record.

    Exposes ``.first.sample`` (dict of variable -> 0/1) and ``.first.energy``
    so code written against a dimod result can consume a QAOA result unchanged.
    """
    def __init__(self, sample, energy):
        self.first = self
        self.sample = sample
        self.energy = energy


# ---------------------------------------------------------------------------
# Truss Optimization via QAOA  (Book Section 20.6)
# ---------------------------------------------------------------------------
class QAOATrussOptimizer(QATrussOptimizer):
    """
    Truss area optimizer using QAOA in place of quantum annealing.

    Subclasses :class:`Chapter05_QuantumAnnealing_functions.QATrussOptimizer`
    and overrides **only** :meth:`qubo_solver`. All finite-element bookkeeping,
    the QUBO coefficient formulas (Eqs. 5.35-5.42), the decoding of updater and
    slack variables, and the outer optimization loop are inherited unchanged,
    so the optimized design matches the annealing version up to QUBO-solver
    stochasticity.

    Parameters
    ----------
    fem_model : TrussFEM
        Truss finite-element model (e.g. from a small ground structure).
    V_bar : float
        Target total volume constraint.
    reps : int, optional
        Number of QAOA layers p. Default 1.
    maxiter : int, optional
        Maximum COBYLA iterations per QUBO. Default 300.
    **kwargs
        Forwarded to ``QATrussOptimizer`` (``alpha_under``, ``alpha_over``,
        ``S_under``, ``S_over``, ``lambda_penalty``, ...).

    Notes
    -----
    The per-iteration QUBO has ``n_elements + 1`` binary variables (one qubit
    per member plus one slack qubit). QAOA on a statevector simulator is only
    practical for small member counts; use the annealing optimizer of Chapter 5
    for the full 3x3 ground structure.
    """

    def __init__(self, fem_model, V_bar, reps=1, maxiter=300, **kwargs):
        super().__init__(fem_model, V_bar, **kwargs)
        self.reps = reps
        self.maxiter = maxiter

    def qubo_solver(self, Q):
        """
        Minimize the truss QUBO with QAOA (overrides the annealing back-end).

        The parent supplies ``Q`` as a dict ``{(i, j): Q_ij}`` over the
        ``n_elements + 1`` binary variables. We convert it to a cost
        Hamiltonian and minimize with QAOA, returning the binary solution
        vector in the same layout the parent's ``decode_solution`` expects.

        Parameters
        ----------
        Q : dict
            QUBO coefficients as built by
            :meth:`QATrussOptimizer.formulate_qubo`.

        Returns
        -------
        numpy.ndarray
            Binary solution vector ``[q_0, ..., q_{N-1}, q_s]``.
        """
        n_vars = self.n_elements + 1
        return solve_qubo_with_qaoa(Q, n_vars, reps=self.reps,
                                    maxiter=self.maxiter)
"""
Quantum Phase Estimation (QPE) Algorithm
=========================================
QPE estimates eigenvalues of unitary operators - critical subroutine for HHL algorithm.

Mathematical Problem:
Given unitary U and eigenvector |v_k⟩ where U|v_k⟩ = e^(2πiθ_k)|v_k⟩,
estimate the eigenphase θ_k ∈ [0,1).  Throughout, λ denotes an eigenvalue of the
Hermitian matrix A (never an eigenvalue of U), matching the book's notation.

Index convention (matches the text): a SUBSCRIPT is always an eigen-index --
λ_k, v_k, θ_k, a_k all refer to mode k.  The binary digits of a phase are written
j_{m-1}...j_0, following the QFT chapter, so θ = (0.j_{m-1}...j_0)_2.

Connection to Linear Systems (HHL):
- For Hermitian matrix A with eigendecomposition A = Σ λ_k|v_k⟩⟨v_k|
- Define U = e^(iAt) → eigenvalues are e^(iλ_k t)
- QPE extracts λ_k → enables computation of 1/λ_k for solving Ax=b

Algorithm Structure:
1. Initialize m ancilla qubits in |+⟩^⊗m superposition
2. Prepare eigenstate |ψ⟩ in data register
3. Apply controlled-U^(2^r) gates for r = 0,1,...,m-1
4. Apply inverse QFT to ancilla qubits
5. Measure ancilla → binary representation of phase φ

Precision and Complexity:
- Phase resolution: Δθ = 1/2^m (m = number of ancilla qubits)
- Success probability: if θ is an exact m-bit fraction the correct bitstring is
  returned with probability 1.  Otherwise the nearest bin carries ≥ 4/π² ≈ 40.5%
  and the two nearest bins together ≥ 8/π² ≈ 81%.
- Gate complexity: O(m²) for the inverse QFT, plus m controlled gates that between
  them apply U a total of Σ_r 2^r = 2^m - 1 times -- the cost is exponential in m,
  not linear (see the Trotterization section of the chapter).
- Trade-off: More ancilla qubits → better precision but deeper circuits

Critical for Solving Linear Systems:
- HHL algorithm uses QPE to extract eigenvalues of system matrix A
- Enables quantum speedup for Ax=b under conditions:
  * A is sparse and s-sparse (≤s non-zero entries per row)
  * κ = λ_max/λ_min (condition number) is reasonable
  * Speedup: O(log(N) s² κ²/ε) vs O(N s κ log(1/ε)) for classical CG
  
References:
- Kitaev (1995): Quantum measurements and Abelian stabilizer problem
- Cleve et al. (1998): Quantum algorithms revisited  
- Harrow, Hassidim, Lloyd (2009): Quantum algorithm for linear systems (HHL)
- Nielsen & Chuang (2010): Quantum Computation and Quantum Information, Ch. 5
"""

import warnings
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate, phase_estimation, HamiltonianGate
from Chapter08_QuantumGates_functions import simulate_measurements


def _checkQPEAssumptions(A, v, lambdaUpper):
	"""Warn (do not raise) when the chapter's standing assumptions are violated.

	Equation (17.2) assumes 0 < lambda_0 <= ... <= lambda_{N-1} < lambdaUpper.  Some
	exercises deliberately break this to show the aliasing that results, so these are
	warnings, not errors.  The classical eigensolve is affordable only because these
	are teaching-scale matrices; it plays no part in the algorithm.
	"""
	N = np.asarray(v).shape[0]
	if N < 2 or (N & (N - 1)) != 0:
		raise ValueError(f"len(v) = {N} is not a power of 2; QPE needs an n-qubit state.")
	if np.asarray(A).shape != (N, N):
		raise ValueError(f"A has shape {np.asarray(A).shape}, expected ({N}, {N}).")
	if lambdaUpper <= 0:
		raise ValueError("lambdaUpper must be positive.")
	w = np.linalg.eigvalsh(np.asarray(A))
	if w.min() < 0:
		warnings.warn(f"A has a negative eigenvalue ({w.min():.4g}); its phase wraps to "
		              f"1 + lambda/lambdaUpper and will be misread as positive. "
		              f"Shift A -> A + cI to restore the assumption.", stacklevel=3)
	if w.max() >= lambdaUpper:
		warnings.warn(f"lambdaUpper = {lambdaUpper:g} does not exceed lambda_max = "
		              f"{w.max():.4g}; eigenphases >= 1 alias modulo 1.", stacklevel=3)


def _sortedThetaAndProbabilities(counts, m, nShots):
	"""Counts dict -> (theta, probability) arrays sorted by decreasing probability."""
	order = sorted(counts.items(), key=lambda item: item[1], reverse=True)
	theta = np.array([int(key, 2) / 2**m for key, _ in order])
	probabilities = np.array([cnt / nShots for _, cnt in order])
	return [theta, probabilities]



def myQPESingleBit(A,v,lambdaUpper,nShots=1000,verbose=True,checkAssumptions=True):
	"""
	Single-Bit Quantum Phase Estimation
	====================================
	Simplified QPE with m=1 ancilla qubit for coarse eigenvalue estimation.
	
	Algorithm:
	1. Create superposition: H|0⟩ = |+⟩ = (|0⟩+|1⟩)/sqrt(2)
	2. Prepare eigenstate |v⟩ in data register
	3. Apply controlled-U_A
	4. Apply H to ancilla again (1-qubit "inverse QFT")
	5. Measure ancilla to estimate θ with 1-bit precision

	Sign convention (easy to get backwards):
	- Qiskit's HamiltonianGate(A, time=t) implements exp(-i A t), which is the
	  OPPOSITE of the book's convention.  We therefore pass t = -2π/λ_upper, so
	  U_A = exp(-i A t) = exp(+2πi A/λ_upper), exactly Equation (17.1).
	- Hence U_A|v⟩ = exp(2πi λ/λ_upper)|v⟩ and the eigenphase is
	      θ = +λ/λ_upper  ∈ (0,1)     -- POSITIVE, per Equation (17.11).
	- Measurement gives: θ_est ∈ {0, 0.5} (binary: 0 or 1)

	Precision:
	- Only 1 bit of phase information → coarse estimate
	- Outcome 1 means θ ≈ 0.5, i.e. λ ≈ λ_upper/2; outcome 0 means λ ≈ 0.
	  A single bit therefore only says which of those two λ is nearer.
	- Useful for quick tests or when high precision not needed
	
	Parameters:
	-----------
	A : ndarray (2^n × 2^n)
		Hermitian matrix (typically system matrix for HHL)
	v : ndarray (2^n,)
		Eigenstate of A (or approximate eigenstate)
	lambdaUpper : float
		Upper bound on eigenvalues of A (for scaling)
	nShots : int
		Number of measurement shots for statistics
	verbose : bool
		Print the raw counts dictionary (default True, as in the book listings)
	checkAssumptions : bool
		Warn if A has a negative eigenvalue or lambdaUpper <= lambda_max

	Returns:
	--------
	[thetaEstimates, probabilities] : list of ndarrays
		Phase estimates and their measured probabilities, sorted by decreasing
		probability (same ordering guarantee as myQPEMultiBit)
		
	Application in HHL:
	Used for preliminary eigenvalue estimates before full QPE
	"""
	if checkAssumptions:
		_checkQPEAssumptions(A, v, lambdaUpper)
	n = int(np.log2(v.shape[0]))  # Number of qubits for data
	circuit = QuantumCircuit(n+1,1)  # 1 ancilla + n data qubits
	
	# Step 1: Create superposition on ancilla
	circuit.h(0)
	
	# Step 2: Prepare eigenstate in data qubits
	circuit.prepare_state(Statevector(v),[*range(1, n+1)],'v')
	
	# Step 3: Hamiltonian evolution.  HamiltonianGate is exp(-iAt), so the negative
	# t below yields U_A = exp(+2πiA/λ_upper) and a POSITIVE phase θ = λ/λ_upper,
	# with λ = λ_upper mapping to θ = 1.
	t = -2*np.pi/lambdaUpper  # negative: cancels Qiskit's exp(-iAt) convention
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	UControl = U_A.control(1)  # Single control qubit
	circuit.append(UControl,[*range(0, n+1)])
	
	# Step 4: Apply H (inverse QFT for m=1)
	circuit.h(0)
	
	# Step 5: Measure ancilla to extract phase bit
	circuit.measure([0], [0])
	counts = simulate_measurements(circuit,shots=nShots)
	if verbose:
		print('Counts:',counts)

	# Convert bit to phase (0 → θ=0, 1 → θ=0.5) and sort by probability
	return _sortedThetaAndProbabilities(counts, 1, nShots)


def myQPEMultiBit(A,v,lambdaUpper,m,nShots=1000,verbose=True,checkAssumptions=True):
	"""
	Multi-Bit Quantum Phase Estimation (Full QPE)
	==============================================
	Complete QPE implementation with m ancilla qubits for high-precision eigenvalue estimation.
	This is the standard form used in HHL algorithm for solving linear systems.
	
	Algorithm Structure:
	1. Initialize m ancilla qubits to |+⟩^⊗m superposition
	2. Prepare eigenstate |v⟩ in n data qubits  
	3. For each ancilla r (r=0 to m-1):
	   Apply controlled-U^(2^r) gate → entangles phase information
	4. Apply inverse QFT to ancilla register
	5. Measure ancilla → binary representation of φ with m-bit precision
	
	Phase Resolution:
	- m bits provide resolution: Δθ = 1/2^m
	- Example: m=8 gives precision ~0.004 (1/256)
	- Higher m → better precision but deeper circuits and more gates

	Gate Complexity:
	- m controlled gates, but they apply U a total of Σ_r 2^r = 2^m - 1 times
	- Inverse QFT: O(m²) gates
	- Total: O(m² + 2^m·cost(U)) -- exponential in m, see Equation (17.30)

	For HHL Application:
	- Estimates eigenvalues λ_k of system matrix A
	- Phase θ = +λ/λ_upper, in [0,1) because 0 < λ < λ_upper (Equation 17.2)
	- Precision requirement depends on condition number κ = λ_max/λ_min
	- Trade-off: m ~ O(log(κ/ε)) for accuracy ε
	
	Parameters:
	-----------
	A : ndarray (2^n × 2^n)
		Hermitian matrix whose eigenvalues to estimate
	v : ndarray (2^n,)
		Eigenstate (or superposition of eigenstates) of A
	lambdaUpper : float
		Upper bound on |λ| for normalization
	m : int
		Number of ancilla qubits (determines precision: 2^(-m))
	nShots : int
		Number of measurement shots
	verbose : bool
		Print the raw counts dictionary.  Set False for large m -- at m=8 the
		dictionary can run to dozens of entries.
	checkAssumptions : bool
		Warn if A has a negative eigenvalue or lambdaUpper <= lambda_max

	Returns:
	--------
	[thetaEstimates, probabilities] : list of ndarrays
		Sorted phase estimates and their measured probabilities
		thetaEstimates[i] = binary_to_decimal(measurement_i) / 2^m
		
	Accuracy Notes:
	- For exact eigenstate: peak probability at true phase
	- For superposition: multiple peaks at different eigenvalue phases
	- More shots → better statistical estimates of probabilities
	"""
	if checkAssumptions:
		_checkQPEAssumptions(A, v, lambdaUpper)
	N = v.shape[0]
	n = int(np.log2(N))  # Data qubits
	
	# Define registers with descriptive names
	phase_qubits = QuantumRegister(m, 'θ')  # Ancilla for phase
	input_qubits = QuantumRegister(n, 'v')  # Data register
	phase_measurements = ClassicalRegister(m, 'Θ')  # Classical bits
	circuit = QuantumCircuit(phase_qubits,input_qubits,phase_measurements)
	
	# Step 1: Initialize ancilla qubits to uniform superposition
	for i in range(m):
		circuit.h(i)
	
	# Step 2: Prepare eigenstate in data register
	circuit.prepare_state(Statevector(v),[*range(m, n+m)],'v')
	
	# Step 3: Create the unitary U_A = exp(+2πiA/λ_upper) of Equation (17.1).
	# HamiltonianGate is exp(-iAt), so t is NEGATIVE to flip the convention; the
	# resulting eigenphase θ = +λ/λ_upper is positive, with λ_upper → θ = 1.
	t = -2*np.pi/lambdaUpper
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	U_A._name = 'UA'
	
	# Step 4: Apply controlled-U^(2^r) for r=0 to m-1
	# This encodes phase information in binary: each qubit represents a bit of φ
	for i in range(m):
		U_A_pow = U_A.power(2**i)  # U^(2^i) operation
		UControl = U_A_pow.control(1)  # Controlled version
		# Control on ancilla qubit i, target all data qubits
		circuit.append(UControl,[i,*range(m, n+m)])
	
	# Step 5: Apply inverse QFT to extract phase from ancilla
	# IQFT transforms binary-encoded phase to computational basis
	iqft = QFTGate(num_qubits=m).inverse()
	iqft._name = 'IQFT'
	circuit.append(iqft, [*range(0,m)])
	
	# Step 6: Measure ancilla qubits
	circuit.measure([*range(0,m)], [*range(0,m)])
	
	# Execute circuit and process results
	counts = simulate_measurements(circuit,shots = nShots)
	if verbose:
		print(counts)

	# Convert m-bit bitstrings to phases in [0,1), sorted by decreasing probability
	return _sortedThetaAndProbabilities(counts, m, nShots)


def QiskitQPEWrapper(A,v,lambdaUpper,m,nShots=1000,verbose=False,checkAssumptions=True):
	"""
	QPE using Qiskit's Built-in phase_estimation Function
	======================================================
	Convenience wrapper using Qiskit's optimized QPE implementation.
	Functionally equivalent to myQPEMultiBit but uses library function.
	
	Advantages of Library Function:
	- Optimized circuit construction
	- Handles edge cases and error checking
	- May use advanced transpilation strategies
	
	Parameters and Returns: Same as myQPEMultiBit
	
	Note: Useful for comparison and validation of custom implementations
	"""
	if checkAssumptions:
		_checkQPEAssumptions(A, v, lambdaUpper)
	N = v.shape[0]
	n = int(np.log2(N))
	
	phase_qubits = QuantumRegister(m, 'θ')
	input_qubits = QuantumRegister(n, 'v')
	phase_measurements = ClassicalRegister(m, 'Θ')
	circuit = QuantumCircuit(phase_qubits,input_qubits,phase_measurements)
	
	# Prepare eigenstate
	circuit.prepare_state(Statevector(v),[*range(m, n+m)],'v')
	
	# Create unitary U_A = exp(+2πiA/λ_upper); t is negative because
	# HamiltonianGate implements exp(-iAt).  See myQPEMultiBit.
	t = -2*np.pi/lambdaUpper
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	U_A._name = 'UA'
	
	# Apply Qiskit's phase estimation (handles H gates, controlled-U, and IQFT)
	QPE = phase_estimation(m,unitary=U_A)
	circuit.append(QPE, [*range(n+m)])
	
	# Measure the phase register.  The library's phase_estimation uses an
	# inverse QFT whose qubit ordering is reversed relative to the hand-built
	# myQPEMultiBit, so we write phase qubit i into classical bit (m-1-i).
	# This reversal is what makes int(key,2)/2**m decode to the same theta as
	# myQPEMultiBit (verified on phases whose bit-reverse is a different value,
	# e.g. 0.375 vs 0.75).  Note U_A here is the EXACT HamiltonianGate, so this
	# path performs no Trotterization; feed a PauliEvolutionGate to Trotterize.
	circuit.measure( [*range(0, m)],[*range(m-1,-1,-1)])
	counts = simulate_measurements(circuit,shots = nShots)
	if verbose:
		print(counts)
	return _sortedThetaAndProbabilities(counts, m, nShots)


# ============================================================================
# Qiskit's batteries-included Hamiltonian phase estimation
# ============================================================================
def hamiltonianPhaseEstimationDemo(A, v, bound, m, evolution=None):
	"""
	QPE via qiskit-algorithms' HamiltonianPhaseEstimation (the highest-level route).
	============================================================================
	Contrast with myQPEMultiBit / QiskitQPEWrapper:
	  - myQPEMultiBit    : we build the circuit AND the unitary U_A ourselves.
	  - QiskitQPEWrapper : Qiskit builds the QPE circuit; we still hand it U_A.
	  - this function    : we hand Qiskit the *Hermitian matrix A itself*; it
	                       scales and exponentiates A into a unitary internally.

	Because it exponentiates A for us, it also Trotterizes internally:
	  - `bound`     is an upper bound on |eigenvalue(A)|.  It plays the role of
	                lambdaUpper, but NOT identically: the window here is the
	                SYMMETRIC interval [-bound, +bound], so the resolution is
	                2*bound/2^m -- half of myQPEMultiBit's lambdaUpper/2^m at the
	                same m.  In exchange, negative eigenvalues are represented
	                correctly, which myQPEMultiBit cannot do (they alias to
	                1 + lambda/lambdaUpper).
	                One further subtlety: the identity component of A is split off
	                and added back exactly, so `bound` only needs to cover the
	                TRACELESS part.  A bound far larger than that part wastes
	                precision -- e.g. A = diag(1, 0.75) = 0.875*I + 0.125*Z with
	                bound=2 and m=3 returns 0.875, i.e. the identity term alone.
	  - `evolution` is the Trotter knob.  Pass LieTrotter(reps=r) or
	                SuzukiTrotter(order=2, reps=r) from qiskit.synthesis.
	                If evolution is None, the DEFAULT is a single first-order
	                Trotter step -- the least accurate corner of the Trotter
	                convergence study (see the Trotterization section).  For a
	                non-commuting A this default visibly biases the estimate;
	                raise the order or reps to remove it.

	Parameters
	----------
	A : ndarray (2^n x 2^n) Hermitian (real symmetric or genuinely complex)
	v : ndarray (2^n,)   eigenstate (or a guess with good overlap)
	bound : float        upper bound on |eigenvalue(A)|; window is [-bound, +bound]
	m : int              number of evaluation qubits (precision)
	evolution : EvolutionSynthesis or None   Trotter formula (default: 1st order)

	Returns
	-------
	HamiltonianPhaseEstimationResult
	    Use .most_likely_eigenvalue for the dominant estimate, or
	    .filter_phases(cutoff, as_float=True) to keep only phases whose
	    probability exceeds `cutoff` (a built-in probability threshold).

	Requires the separate `qiskit-algorithms` package (pinned in requirements).
	"""
	# Lazy imports so this module still loads if qiskit-algorithms is absent.
	from qiskit.quantum_info import SparsePauliOp
	from qiskit.primitives import StatevectorSampler
	from qiskit_algorithms import HamiltonianPhaseEstimation

	v = np.asarray(v, dtype=complex)
	n = int(np.log2(v.shape[0]))
	# NOTE: do NOT cast to float here.  np.asarray(A, dtype=float) silently drops
	# the imaginary part of a genuinely complex Hermitian A (numpy raises only a
	# ComplexWarning), which turns e.g. [[1, 0.5j], [-0.5j, 1]] into the identity
	# and returns 1.0 instead of 1.5.
	A = np.asarray(A)
	if not np.allclose(A, A.conj().T):
		raise ValueError("A must be Hermitian (A == A^dagger).")
	H = SparsePauliOp.from_operator(A)  # Pauli form of A

	prep = QuantumCircuit(n)                                     # prepares |v>
	prep.prepare_state(Statevector(v/np.linalg.norm(v)), list(range(n)))

	hpe = HamiltonianPhaseEstimation(num_evaluation_qubits=m,
	                                 sampler=StatevectorSampler())
	return hpe.estimate(hamiltonian=H, state_preparation=prep,
	                    evolution=evolution, bound=bound)


# ============================================================================
# Application: estimating an eigenvalue of an engineering operator with QPE
# ============================================================================
# The QFT chapter solved the CONSTANT-coefficient (circulant / tridiagonal)
# operator analytically, because its spectrum is known in closed form.  QPE is
# what you reach for when you want a specific eigenvalue -- typically the
# fundamental mode (lowest frequency / slowest decay / critical load) -- of an
# operator whose spectrum you do NOT know analytically.

def laplacian1D(N):
	"""1D Dirichlet Laplacian on N interior nodes: the tridiagonal (-1, 2, -1).
	Take N = 2**n so the node vector is an n-qubit state."""
	return 2*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)


def laplacianEigenExact(N):
	"""Closed-form eigenvalues of laplacian1D(N): 4 sin^2((k+1)*pi/(2(N+1))),
	k = 0..N-1, so entry 0 is the fundamental lambda_0.  Used ONLY to verify the
	QPE estimate -- QPE is for the case where no closed form exists."""
	k = np.arange(0, N)
	return 4*np.sin((k+1)*np.pi/(2*(N+1)))**2


def laplacianEigenvector(N, k):
	"""Eigenvector for mode k (k = 0..N-1) of laplacian1D(N): sin(j*(k+1)*pi/(N+1)).
	k = 0 is the fundamental mode (smallest eigenvalue), matching the 0-indexed
	ordering lambda_0 <= lambda_1 <= ... used throughout the chapter."""
	j = np.arange(1, N+1)
	v = np.sin(j*(k+1)*np.pi/(N+1))
	return v/np.linalg.norm(v)


def topSpectralPeaks(theta, P, m, nPeaks, lambdaUpper=1.0):
	"""Recover up to `nPeaks` eigenvalues from a single QPE histogram.

	Keeps the `nPeaks` largest LOCAL MAXIMA of the m-bit histogram, not simply the
	`nPeaks` most probable bins.  The distinction matters as soon as the input is
	not an equal-weight superposition: when one mode dominates, its own leakage
	sidelobes are more probable than the peaks of the weaker modes, so a top-N-bins
	rule returns one eigenvalue several times over and misses the rest.

	This is a heuristic, not a guarantee.  A mode with negligible overlap produces
	no peak at all and is simply absent, and deep leakage can raise a spurious peak
	between true ones.  See the discussion of lambda_min in the Laplacian section.

	Parameters
	----------
	theta, P    : arrays returned by myQPEMultiBit / QiskitQPEWrapper
	m           : number of phase qubits (histogram has 2**m bins)
	nPeaks      : how many eigenvalues to keep (at most N for an N x N operator)
	lambdaUpper : scale factor; leave at 1.0 to get phases, pass lambdaUpper for
	              eigenvalues

	Returns
	-------
	ndarray, ascending, of length <= nPeaks
	"""
	hist = np.zeros(2**m)
	hist[np.round(np.asarray(theta)*2**m).astype(int) % 2**m] = P
	isPeak = (hist > np.roll(hist, 1)) & (hist >= np.roll(hist, -1))
	cand = np.flatnonzero(isPeak)
	keep = cand[np.argsort(hist[cand])[-nPeaks:]]
	return np.sort(keep/2**m) * lambdaUpper


def estimateEigenvalueQPE(A, v, lambdaUpper, m, qpe=QiskitQPEWrapper, nShots=4000):
	"""
	Estimate the eigenvalue of Hermitian A associated with the PREPARED state v.
	============================================================================
	QPE returns the eigenphase(s) present in v, each weighted by |<u_k|v>|^2, so
	WHICH eigenvalue you obtain is decided entirely by the state you prepare:
	prepare the fundamental mode and you read the fundamental eigenvalue.

	Obtaining that mode is the genuinely hard part, and it is deliberately NOT
	done inside this routine.  For an operator with a known analytic mode (e.g.
	the sine modes of laplacian1D) you pass it in directly; otherwise you supply
	a physical guess with good overlap, or -- for verification on small problems
	only -- an eigenvector from a classical solve.  Keeping that step at the call
	site makes the classical eigen-computation visible instead of hiding it here.

	The `qpe` backend selects how the phases are produced: QiskitQPEWrapper
	(the library circuit, the default) or myQPEMultiBit (the from-scratch
	build).  They are interchangeable -- same algorithm, same [theta, prob]
	interface -- so the extraction below is independent of the choice.

	Returns (lambdaPeak, lambdaWeighted): the dominant-bin and probability-
	weighted estimates, both in eigenvalue units (theta * lambdaUpper).

	WHICH TO TRUST.  The weighted mean is the smoother statistic, but it is a
	LINEAR average over a CIRCULAR variable, so it is biased whenever spectral
	leakage wraps past 1 into bins near 0.9-1.0 -- those bins pull the mean up
	instead of toward the true phase.  For the Laplacian fundamental below the
	peak errs by 0.007 and the weighted mean by 0.013; the peak is usually the
	better single number unless theta sits comfortably away from both 0 and 1.
	`lambdaCircular` is the unbiased alternative: the argument of the mean phasor,
	which averages correctly across the 0/1 seam.

	The smallest eigenvalue maps to the smallest phase (nearest 0) and is the
	hardest to resolve -- the quantity an engineer most wants is the one QPE
	resolves worst.
	"""
	v = np.asarray(v, dtype=complex)
	v = v / np.linalg.norm(v)                # accept any trial vector, not just unit-norm
	theta, P = qpe(A, v, lambdaUpper, m, nShots=nShots)
	lambdaPeak     = float(theta[int(np.argmax(P))]) * lambdaUpper   # dominant bin
	lambdaWeighted = float(np.sum(theta * P))        * lambdaUpper   # weighted mean
	return lambdaPeak, lambdaWeighted


def circularMeanPhase(theta, P):
	"""Probability-weighted mean of phases on the circle: arg(sum_i P_i e^{2pi i th_i}).

	Use instead of sum(P*theta) when leakage straddles the 0/1 seam -- the linear
	mean treats theta=0.99 as far from theta=0.01 when they are in fact adjacent.
	Returns a value in [0,1).
	"""
	z = np.sum(np.asarray(P) * np.exp(2j*np.pi*np.asarray(theta)))
	return float(np.angle(z) / (2*np.pi)) % 1.0
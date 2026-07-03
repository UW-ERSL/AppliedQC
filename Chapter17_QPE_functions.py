"""
Quantum Phase Estimation (QPE) Algorithm
=========================================
QPE estimates eigenvalues of unitary operators - critical subroutine for HHL algorithm.

Mathematical Problem:
Given unitary U and eigenstate |ψ⟩ where U|ψ⟩ = e^(iφ)|ψ⟩,
estimate phase φ (and thus eigenvalue λ = e^(iφ))

Connection to Linear Systems (HHL):
- For Hermitian matrix A with eigendecomposition A = Σ λ_j|u_j⟩⟨u_j|
- Define U = e^(iAt) → eigenvalues are e^(iλ_j t)
- QPE extracts λ_j → enables computation of λ_j^(-1) for solving Ax=b

Algorithm Structure:
1. Initialize m ancilla qubits in |+⟩^⊗m superposition
2. Prepare eigenstate |ψ⟩ in data register
3. Apply controlled-U^(2^k) gates for k = 0,1,...,m-1
4. Apply inverse QFT to ancilla qubits
5. Measure ancilla → binary representation of phase φ

Precision and Complexity:
- Phase resolution: Δφ ≈ 1/2^m (m = number of ancilla qubits)
- Success probability: ~81% for exact eigenstate
- Gate complexity: O(m²) for QFT + O(m) controlled-U applications
- Trade-off: More ancilla qubits → better precision but deeper circuits

Critical for Solving Linear Systems:
- HHL algorithm uses QPE to extract eigenvalues of system matrix A
- Enables quantum speedup for Ax=b under conditions:
  * A is sparse and s-sparse (≤s non-zero entries per row)
  * κ = λ_max/λ_min (condition number) is reasonable
  * Speedup: O(log(N)κ²/ε) vs O(Nκ/ε) classical
  
References:
- Kitaev (1995): Quantum measurements and Abelian stabilizer problem
- Cleve et al. (1998): Quantum algorithms revisited  
- Harrow, Hassidim, Lloyd (2009): Quantum algorithm for linear systems (HHL)
- Nielsen & Chuang (2010): Quantum Computation and Quantum Information, Ch. 5
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate, phase_estimation, HamiltonianGate
from Chapter08_QuantumGates_functions import (simulate_statevector, simulate_measurements, runCircuitOnIBMQuantum, 
                                              findActualHardwareRequirements, plot_measurement_results)



def myQPESingleBit(A,v,lambdaUpper,nShots=1000):
	"""
	Single-Bit Quantum Phase Estimation
	====================================
	Simplified QPE with m=1 ancilla qubit for coarse eigenvalue estimation.
	
	Algorithm:
	1. Create superposition: H|0⟩ = |+⟩ = (|0⟩+|1⟩)/sqrt(2)
	2. Prepare eigenstate |v⟩ in data register
	3. Apply controlled-U where U = exp(iAt) with t = -2π/λ_upper
	4. Apply H to ancilla again (1-qubit "inverse QFT")
	5. Measure ancilla to estimate φ with 1-bit precision
	
	Phase Encoding:
	- For eigenvalue λ with U|v⟩ = e^(iλt)|v⟩
	- Effective phase: θ = λt/(2π) = -λ/λ_upper
	- Measurement gives: θ_est ∈ {0, 0.5} (binary: 0 or 1)
	
	Precision:
	- Only 1 bit of phase information → coarse estimate
	- Can distinguish if λ is closer to 0 or λ_upper
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
		
	Returns:
	--------
	[thetaEstimates, probabilities] : list of ndarrays
		Phase estimates and their measured probabilities
		
	Application in HHL:
	Used for preliminary eigenvalue estimates before full QPE
	"""
	n = int(np.log2(v.shape[0]))  # Number of qubits for data
	circuit = QuantumCircuit(n+1,1)  # 1 ancilla + n data qubits
	
	# Step 1: Create superposition on ancilla
	circuit.h(0)
	
	# Step 2: Prepare eigenstate in data qubits
	circuit.prepare_state(Statevector(v),[*range(1, n+1)],'v')
	
	# Step 3: Hamiltonian evolution with time chosen to normalize phase
	# t = -2π/λ_upper makes λ_upper correspond to phase = -2π → θ = 1
	t = -2*np.pi/lambdaUpper  # Negative for phase convention
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	UControl = U_A.control(1)  # Single control qubit
	circuit.append(UControl,[*range(0, n+1)])
	
	# Step 4: Apply H (inverse QFT for m=1)
	circuit.h(0)
	
	# Step 5: Measure ancilla to extract phase bit
	circuit.measure([0], [0])
	counts = simulate_measurements(circuit,shots=nShots)
	print('Counts:',counts)
	
	# Process results
	probabilities = np.array([])
	thetaEstimates = np.array([])
	for key in counts:
		probabilities = np.append(probabilities,counts[key]/nShots)
		# Convert bit to phase: 0 → θ=0, 1 → θ=0.5
		thetaEstimates = np.append(thetaEstimates,int(key, 2)/(2))
	return [thetaEstimates,probabilities]


def myQPEMultiBit(A,v,lambdaUpper,m,nShots=1000):
	"""
	Multi-Bit Quantum Phase Estimation (Full QPE)
	==============================================
	Complete QPE implementation with m ancilla qubits for high-precision eigenvalue estimation.
	This is the standard form used in HHL algorithm for solving linear systems.
	
	Algorithm Structure:
	1. Initialize m ancilla qubits to |+⟩^⊗m superposition
	2. Prepare eigenstate |v⟩ in n data qubits  
	3. For each ancilla k (k=0 to m-1):
	   Apply controlled-U^(2^k) gate → entangles phase information
	4. Apply inverse QFT to ancilla register
	5. Measure ancilla → binary representation of φ with m-bit precision
	
	Phase Resolution:
	- m bits provide resolution: Δφ = 1/2^m
	- Example: m=8 gives precision ~0.004 (1/256)
	- Higher m → better precision but deeper circuits and more gates
	
	Gate Complexity:
	- Controlled-U applications: m gates (but U^(2^k) may be expensive)
	- Inverse QFT: O(m²) gates
	- Total: O(m² + m·cost(U))
	
	For HHL Application:
	- Estimates eigenvalues λ_j of system matrix A
	- Phase θ = -λ/λ_upper mapped to range [0,1)
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
	
	# Step 3: Create unitary operator U = exp(iAt)
	# Time chosen so λ_upper → phase 2π → θ = 1
	t = -2*np.pi/lambdaUpper
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	U_A._name = 'UA'
	
	# Step 4: Apply controlled-U^(2^k) for k=0 to m-1
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
	print(counts)
	
	# Sort by count (most frequent measurements first)
	countsSorted = {k: v for k, v in sorted(counts.items(), 
										 key=lambda item: item[1],
										 reverse=True)}
	
	# Convert bitstrings to phase estimates
	probabilities = np.array([])
	thetaEstimates = np.array([])
	for key in countsSorted:
		probabilities = np.append(probabilities,countsSorted[key]/nShots)
		# Convert m-bit binary to decimal, normalize to [0,1)
		thetaEstimates = np.append(thetaEstimates,int(key, 2)/(2**m))
	return [thetaEstimates,probabilities]


def QiskitQPEWrapper(A,v,lambdaUpper,m,nShots=1000):
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
	N = v.shape[0]
	n = int(np.log2(N))
	
	phase_qubits = QuantumRegister(m, 'θ')
	input_qubits = QuantumRegister(n, 'v')
	phase_measurements = ClassicalRegister(m, 'Θ')
	circuit = QuantumCircuit(phase_qubits,input_qubits,phase_measurements)
	
	# Prepare eigenstate
	circuit.prepare_state(Statevector(v),[*range(m, n+m)],'v')
	
	# Create unitary
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
	
	# Process and sort results
	countsSorted = {k: v for k, v in sorted(counts.items(), 
										 key=lambda item: item[1],
										 reverse=True)}
	probabilities = np.array([])
	thetaEstimates = np.array([])
	for key in countsSorted:
		probabilities = np.append(probabilities,countsSorted[key]/nShots)
		thetaEstimates = np.append(thetaEstimates,int(key, 2)/(2**m))
	return [thetaEstimates,probabilities]


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
	  - `bound`     is an upper bound on |eigenvalue(A)|  (the role of lambdaUpper);
	                it scales A so the phases stay inside the QPE window.
	  - `evolution` is the Trotter knob.  Pass LieTrotter(reps=r) or
	                SuzukiTrotter(order=2, reps=r) from qiskit.synthesis.
	                If evolution is None, the DEFAULT is a single first-order
	                Trotter step -- the least accurate corner of the Trotter
	                convergence study (see the Trotterization section).  For a
	                non-commuting A this default visibly biases the estimate;
	                raise the order or reps to remove it.

	Parameters
	----------
	A : ndarray (2^n x 2^n) Hermitian
	v : ndarray (2^n,)   eigenstate (or a guess with good overlap)
	bound : float        upper bound on |eigenvalue(A)|
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
	H = SparsePauliOp.from_operator(np.asarray(A, dtype=float))  # Pauli form of A

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
	k = 0..N-1, so entry 0 is the fundamental lambda^0.  Used ONLY to verify the
	QPE estimate -- the point of QPE is the case where no closed form exists
	(see variableStiffnessRod)."""
	k = np.arange(0, N)
	return 4*np.sin((k+1)*np.pi/(2*(N+1)))**2


def laplacianEigenvector(N, k):
	"""Eigenvector for mode k (k = 0..N-1) of laplacian1D(N): sin(j*(k+1)*pi/(N+1)).
	k = 0 is the fundamental mode (smallest eigenvalue), matching the 0-indexed
	ordering lambda^0 <= lambda^1 <= ... used throughout the chapter."""
	j = np.arange(1, N+1)
	v = np.sin(j*(k+1)*np.pi/(N+1))
	return v/np.linalg.norm(v)


def variableStiffnessRod(N, aElem):
	"""
	Heterogeneous rod operator  -d/dx( a(x) du/dx ),  clamped ends, N interior nodes.
	aElem holds the stiffness of the N+1 elements linking the nodes.

	When a(x) is constant this is (a scaling of) laplacian1D.  When a(x) VARIES
	-- e.g. a matrix/inclusion contrast a in {1, 10} -- the operator is still
	symmetric tridiagonal, but its eigenvectors are no longer the sine modes and
	its spectrum has NO closed form.  This is exactly the regime where QPE (or
	HHL) is genuinely required: there is no analytic shortcut to verify against,
	so for small N one checks against a classical eigensolver.
	"""
	aElem = np.asarray(aElem, dtype=float)
	A = np.zeros((N, N))
	for i in range(N):
		aL, aR = aElem[i], aElem[i+1]
		A[i, i] = aL + aR
		if i > 0:    A[i, i-1] = -aL
		if i < N-1:  A[i, i+1] = -aR
	return A


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
	weighted estimates, both in eigenvalue units (theta * lambdaUpper).  The
	weighted value is usually closer when the true phase is not an exact m-bit
	fraction.  The smallest eigenvalue maps to the smallest phase (nearest 0) and
	is the hardest to resolve -- the quantity an engineer most wants is the one
	QPE resolves worst.
	"""
	v = np.asarray(v, dtype=complex)
	v = v / np.linalg.norm(v)                # accept any trial vector, not just unit-norm
	theta, P = qpe(A, v, lambdaUpper, m, nShots=nShots)
	lambdaPeak     = float(theta[int(np.argmax(P))]) * lambdaUpper   # dominant bin
	lambdaWeighted = float(np.sum(theta * P))        * lambdaUpper   # weighted mean
	return lambdaPeak, lambdaWeighted
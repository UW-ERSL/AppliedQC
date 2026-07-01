"""
Quantum Fourier Transform (QFT) Implementation
===============================================
QFT is the quantum analog of the discrete Fourier transform (DFT).
Foundation for QPE, HHL (linear systems solver), and Shor's algorithm.

Key Mathematical Transformation:
|j⟩ → (1/sqrt(2^n)) Σ_{k=0}^{2^n-1} exp(2πijk/2^n)|k⟩

Computational Complexity:
- Classical DFT: O(N log N) with FFT (Fast Fourier Transform)
- Quantum QFT: O(n²) gates where n = log₂(N), i.e., O(log²N)
- Exponential speedup in circuit depth, but measurement loses information

Structure:
- Hadamard gates on each qubit
- Controlled phase rotations: CR_k(θ) where θ = 2π/2^k
- SWAP gates for bit reversal

Applications in Linear Systems:
- HHL algorithm uses QFT for eigenvalue estimation
- Enables quantum speedup for solving Ax=b under certain conditions
- Relevant for design optimization problems with large sparse matrices

References:
- Coppersmith (1994): Approximate Fourier transform useful in quantum factoring
- Nielsen & Chuang (2010): Quantum Computation and Quantum Information, Ch. 5
- Harrow, Hassidim, Lloyd (2009): Quantum algorithm for linear systems (uses QFT in QPE)
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate
from qiskit.circuit.library import QFTGate, UCRYGate, UnitaryGate
from Chapter08_QuantumGates_functions import (simulate_statevector, simulate_measurements, runCircuitOnIBMQuantum, 
                                              findActualHardwareRequirements, plot_measurement_results)


def trignometricSignal(t,c = [],s = []):
	"""
	Generate trigonometric signal from Fourier coefficients
	========================================================
	Constructs time-domain signal: f(t) = Σ c_k·cos(2πkt) + Σ s_k·sin(2πkt)
	
	Parameters:
	-----------
	t : float or array
		Time point(s) in [0, 1)
	c : list
		Cosine coefficients [c_0, c_1, ..., c_{M/2-1}]
	s : list
		Sine coefficients [s_0, s_1, ..., s_{M/2-1}]
		
	Returns:
	--------
	signal : float or array
		Signal value(s) at time t
	"""
	signal = sum(ck * np.cos(k * 2 * np.pi * t) for k, ck in enumerate(c)) + \
		sum(sk * np.sin(k * 2 * np.pi * t) for k, sk in enumerate(s))
	return signal

def createDFTMatrix(M):
	"""
	Create Discrete Fourier Transform matrix
	=========================================
	DFT matrix F where F[j,k] = ω^(jk) and ω = exp(-2πi/M)
	Classical DFT: y = F·x with O(M²) complexity (naive) or O(M log M) with FFT
	
	Parameters:
	-----------
	M : int
		Size of DFT (number of samples)
		
	Returns:
	--------
	F : ndarray (M × M)
		Complex DFT matrix
		
	Note: DFT and QFT differ by normalization and sign convention
	"""
	omega = np.exp(-2j * np.pi / M)
	i, j = np.meshgrid(np.arange(M), np.arange(M))
	return omega ** (i * j)/M

def processDFTResult(phi):
	"""
	Extract real Fourier coefficients from DFT output
	==================================================
	Converts complex DFT spectrum to real cosine/sine coefficients.
	Uses symmetry: phi[M-k] = conj(phi[k]) for real signals
	
	Parameters:
	-----------
	phi : ndarray (M,)
		Complex DFT output
		
	Returns:
	--------
	[c, s] : list of ndarrays
		c: Cosine coefficients (M/2,)
		s: Sine coefficients (M/2,)
		
	Mathematical Basis:
	cos(kθ) = (e^(ikθ) + e^(-ikθ))/2
	sin(kθ) = (e^(ikθ) - e^(-ikθ))/(2i)
	"""
	M = len(phi)
	c = np.zeros(int(M/2))
	s = np.zeros(int(M/2))
	c[0] =  phi[0].real  # DC component (constant term)
	c[1:] = (phi[1:int(M/2)]+phi[M-1:int(M/2):-1]).real  # Cosine terms from symmetry
	s =  (phi[M-1:int(M/2):-1] - phi[1:int(M/2)]).imag  # Sine terms from symmetry
	s = np.insert(s, 0,0)  # No DC sine component
	return [c,s]


def plotDFTResult(c,s,M):
	"""
	Visualize Fourier spectrum
	===========================
	Plots cosine and sine coefficients vs frequency index.
	Useful for identifying dominant frequency components.
	
	Parameters:
	-----------
	c : ndarray
		Cosine coefficients
	s : ndarray
		Sine coefficients
	M : int
		Total number of samples
	"""
	plt.figure()
	plt.bar(list(range(0,int(M/2))),c, label =r"$c_k$")
	plt.bar(list(range(0,int(M/2))),s, label =r"$s_k$")
	plt.legend( fontsize=14)
	plt.axhline(0, color='black')
	plt.xlabel('Frequency', fontsize=14)
	plt.ylabel('Amplitude', fontsize=14)

	plt.title('DFT Result', fontsize=16)
	plt.grid(visible=True)
	plt.show()
	
def QFTSignalProcessing(y,shots=1000):
	"""
	Apply Quantum Fourier Transform to signal
	==========================================
	Uses QFT circuit to transform signal from time to frequency domain.
	
	Algorithm Steps:
	1. Encode signal amplitudes in quantum state |ψ⟩ = Σ y_j|j⟩ (normalized)
	2. Apply QFT: |ψ⟩ → (1/sqrt(M)) Σ_k ŷ_k|k⟩
	3. Measure to collapse to frequency basis states
	4. Repeat 'shots' times to estimate probability distribution
	
	Complexity:
	- Classical FFT: O(M log M) operations
	- QFT circuit: O(n²) gates where n = log₂(M)
	- But: Measurement requires O(M) shots to reconstruct full spectrum
	
	Parameters:
	-----------
	y : ndarray (M,)
		Input signal (M must be power of 2)
	shots : int
		Number of measurements for statistics
		
	Returns:
	--------
	counts : dict
		Measurement outcomes {bitstring: count}
		Keys represent frequency indices in binary
	"""
	M = len(y)  # Signal length
	m = int(np.log2(M))  # Number of qubits needed
	print('Number of qubits:',m)
	circuit = QuantumCircuit(m, m)
	
	# Encode signal as quantum state (amplitude encoding)
	q = Statevector(y/np.linalg.norm(y))  # Normalize to unit vector
	circuit.prepare_state(q,list(range(m)),'Prepare q')
	
	# Apply QFT transformation
	qftCircuit = QFTGate(num_qubits=m)
	circuit.append(qftCircuit, qargs=list(range(m)))
	
	# Measure all qubits to extract frequency information
	circuit.measure(list(range(m)),list(range(m)))
	counts = simulate_measurements(circuit,shots=shots)
		
	return counts

def processQFTResult(N,counts,shots):
	"""
	Extract amplitude spectrum from QFT measurement results
	========================================================
	Reconstructs frequency amplitudes from measured bitstring counts.
	
	Process:
	- Each bitstring k represents frequency component |k⟩
	- Count frequency: N_k → Probability: p_k = N_k/shots
	- Amplitude: |ŷ_k| ≈ sqrt(p_k)
	
	Parameters:
	-----------
	N : int
		Signal length (number of frequency bins)
	counts : dict
		Measurement results from QFT circuit
	shots : int
		Total number of measurements
		
	Returns:
	--------
	ampl : ndarray (N/2,)
		Amplitude spectrum (positive frequencies only due to symmetry)
		
	Note: Phase information is lost in measurement; only magnitudes recovered
	"""
	phi = np.zeros(N)
	# Extract amplitudes from measurement counts
	for i in counts:
		freq = int(i, 2)  # Convert bitstring to frequency index
		phi[freq] = np.sqrt(counts[i]/shots)  # Amplitude proportional to sqrt(probability)
	
	# Use symmetry to combine positive and negative frequencies
	ampl = (phi[1:int(N/2)])+(phi[N-1:int(N/2):-1])
	ampl = np.insert(ampl, 0,phi[0].real)  # Add DC component
	return ampl
	
def myQFT(m):
	"""
	Custom QFT circuit implementation
	==================================
	Builds QFT circuit from scratch using Hadamard and controlled phase gates.
	Equivalent to Qiskit's QFTGate but shows explicit structure.
	
	Circuit Structure (for n qubits):
	1. For each qubit k (from n-1 to 0):
	   a. Apply Hadamard to qubit k
	   b. Apply controlled phase rotations: CR_j(2π/2^j) from qubits j<k
	2. Apply SWAP gates to reverse qubit order (bit reversal)
	
	Gate Count:
	- Hadamards: n gates
	- Controlled phases: n(n-1)/2 gates
	- SWAPs: n/2 gates
	Total: O(n²) gates where n = log₂(N)
	
	Parameters:
	-----------
	m : int
		Number of qubits
		
	Returns:
	--------
	circuit : QuantumCircuit
		QFT circuit with m qubits
		
	Efficiency Note:
	This O(n²) gate complexity compares favorably to classical FFT's O(N log N)
	when expressed in terms of problem size N = 2^n: O(log²N) vs O(N log N)
	"""
	q = QuantumRegister(m, 'q')
	c = ClassicalRegister(m,'c')
	circuit = QuantumCircuit(q,c)
	
	# Build QFT: process qubits from most significant to least significant
	for k in range(m):
		kk = m - k  # Index from end (reverse order)
		circuit.h(q[kk-1])  # Hadamard creates superposition
		circuit.barrier()
		
		# Controlled phase rotations: entangle with more significant qubits
		for i in reversed(range(kk-1)):
			# CR_k gate with angle 2π/2^(kk-i)
			circuit.cp(2*np.pi/2**(kk-i),q[i], q[kk-1])
	  
	circuit.barrier()
	# Bit reversal: Swap qubits to correct output order
	for i in range(m//2):
		circuit.swap(q[i], q[m-i-1])
	return circuit

def createQFTMatrix(N):
	"""
	Generate QFT transformation matrix
	===================================
	Creates unitary matrix representation of QFT.
	QFT[j,k] = (1/sqrt(N)) · ω^(jk) where ω = exp(2πi/N)
	
	Difference from DFT:
	- QFT uses positive sign in exponent: exp(+2πi/N)
	- QFT includes 1/sqrt(N) normalization (unitary)
	- DFT uses exp(-2πi/N) without normalization
	
	Parameters:
	-----------
	N : int
		Transform size (must be power of 2 for quantum implementation)
		
	Returns:
	--------
	QFTMatrix : ndarray (N × N)
		Complex unitary QFT matrix
		
	Properties:
	- Unitary: QFT† · QFT = I
	- Inverse: QFT^(-1) = QFT†
	- Eigenvalues all have magnitude 1
	"""
	omega = np.exp(1j*(2*np.pi/N))
	i, j = np.meshgrid(np.arange(N), np.arange(N))
	QFTMatrix = omega ** (i * j)/np.sqrt(N)  # Include normalization
	return QFTMatrix

# =============================================================================
# Spectral Poisson solver  --  addendum to Chapter16_QFT_functions.py
# -----------------------------------------------------------------------------
# Companion code for Section "Application: A Spectral Poisson Solver".
# =============================================================================

def poissonStiffness(N):
	"""
	1-D Dirichlet Poisson stiffness matrix
	=======================================
	N x N tridiagonal matrix tridiag(-1, 2, -1) from central finite differences
	for -u'' = f on [0, 1] with u(0)=u(1)=0, boundary nodes eliminated.
	"""
	return (np.diag(2.0*np.ones(N))
	        - np.diag(np.ones(N-1), 1)
	        - np.diag(np.ones(N-1), -1))


def poissonEigen(N):
	"""
	Closed-form eigenpairs of the 1-D Poisson matrix
	=================================================
	    lambda_k = 4 sin^2( k*pi / (2(N+1)) ),    k = 1..N
	    v_k(j)   = sqrt(2/(N+1)) sin( j*k*pi/(N+1) )
	The eigenvectors are the discrete sine modes; stacked as columns they form
	the discrete sine transform S (real, symmetric, orthogonal: S = S^T = S^-1),
	which diagonalizes A = S Lambda S.

	Returns
	-------
	lam : ndarray (N,)    eigenvalues, ordered k = 1..N
	S   : ndarray (N, N)  sine-transform matrix (columns are eigenvectors)
	"""
	k = np.arange(1, N+1)
	lam = 4.0*np.sin(k*np.pi/(2*(N+1)))**2
	j = np.arange(1, N+1)
	S = np.sqrt(2.0/(N+1))*np.sin(np.outer(j, k)*np.pi/(N+1))
	return lam, S


def sineTransformViaDFT(x):
	"""
	Discrete sine transform realized from a DFT of the odd extension
	================================================================
	Shows that the sine transform is "the QFT up to an odd-extension embedding":
	the length-N DST equals -1/2 the imaginary part of the length-2(N+1) DFT of
	the odd extension of x. Classical mirror of the quantum sine transform.
	"""
	N = len(x)
	xtilde = np.concatenate([[0.0], x, [0.0], -x[::-1]])
	X = np.fft.fft(xtilde)
	return -0.5*np.imag(X[1:N+1])


def spectralPoissonCircuit(f, measure=True):
	"""
	Build the spectral-Poisson compliance circuit
	==============================================
	On m = log2(N) system qubits plus one ancilla:
	  (1) encode the normalized load |psi_b> ,
	  (2) apply the sine transform U_S (eigenbasis of A),
	  (3) invert eigenvalues with a multiplexed Ry that rotates the ancilla by
	      theta_k with sin(theta_k/2) = sqrt(lam_min/lam_k) for each basis |k>.
	Step (3) needs NO phase estimation: lam_k is a closed-form function of the
	basis label k, so the angle is computed directly.

	Parameters
	----------
	f : ndarray (N,)   load samples at interior nodes (N a power of 2)
	measure : bool     append a measurement on the ancilla

	Returns
	-------
	circuit : QuantumCircuit
	lam_min : float    smallest eigenvalue (used to rescale the result)
	"""
	N = len(f)
	m = int(np.log2(N))
	if 2**m != N:
		raise ValueError("len(f) must be a power of 2")

	lam, S = poissonEigen(N)
	lam_min = lam.min()
	angles = 2.0*np.arcsin(np.sqrt(lam_min/lam))   # closed-form eigenvalue inversion

	q = QuantumRegister(m, 'q')      # system register
	a = QuantumRegister(1, 'a')      # ancilla for eigenvalue inversion
	circuit = QuantumCircuit(q, a)

	# (1) encode the normalized load |psi_b>
	psi = f/np.linalg.norm(f)
	if np.allclose(psi, np.ones(N)/np.sqrt(N)):
		circuit.h(q)                                       # uniform load: O(m) Hadamards
	else:
		circuit.prepare_state(Statevector(psi), list(q))   # general load: O(2^m)

	# (2) sine transform (QFT up to the odd-extension embedding)
	circuit.append(UnitaryGate(S, label='S'), list(q))

	# (3) eigenvalue inversion via a uniformly controlled (multiplexed) Ry
	circuit.append(UCRYGate(list(angles)), [a[0]] + list(q))

	if measure:
		c = ClassicalRegister(1, 'c')
		circuit.add_register(c)
		circuit.measure(a[0], c[0])
	return circuit, lam_min


def complianceFromProbability(f, P, lam_min):
	"""
	Rescale the ancilla probability into the compliance
	====================================================
	P = lam_min * <psi_b| A^-1 |psi_b>. With the quadrature weight that makes
	the discrete energy converge to the continuum value,
	    c = h^3 (f.f) P / lam_min,    h = 1/(N+1).
	"""
	N = len(f)
	h = 1.0/(N+1)
	return h**3*(f @ f)*P/lam_min


def spectralComplianceQuantum(f, shots=100000):
	"""
	Compliance via the quantum pipeline (sampling stands in for QAE)
	================================================================
	Sampling P costs O(1/eps^2); quantum amplitude estimation (Chapter 15)
	reduces this to O(1/eps). Returns (compliance, P_hat).
	"""
	circuit, lam_min = spectralPoissonCircuit(f, measure=True)
	counts = simulate_measurements(circuit, shots=shots)
	P = counts.get('1', 0)/shots
	return complianceFromProbability(f, P, lam_min), P


def spectralComplianceExact(f):
	"""
	Statevector-exact compliance through the same circuit (verification)
	====================================================================
	"""
	circuit, lam_min = spectralPoissonCircuit(f, measure=False)
	m = int(np.log2(len(f)))
	P = float(Statevector(circuit).probabilities([m])[1])
	return complianceFromProbability(f, P, lam_min), P


def spectralComplianceClassical(f):
	"""
	Reference compliance  c = h^3 f^T A^-1 f  (classical check)
	===========================================================
	"""
	N = len(f)
	h = 1.0/(N+1)
	A = poissonStiffness(N)
	return h**3*(f @ np.linalg.solve(A, f))




# ---------------------------------------------------------------------------
# Periodic spectral Poisson solver (append to Chapter16_QFT_functions.py).
# Reuses QFTGate, UCRYGate, Statevector, simulate_measurements already imported.
# ---------------------------------------------------------------------------

def periodicPoissonStiffness(N):
	"""
	Circulant 1-D periodic Poisson matrix
	======================================
	tridiag(-1, 2, -1) with the two wrap-around corners set to -1, modeling a
	ring (Chapter 3). Symmetric and singular: the constant vector is a null mode.
	"""
	C = 2.0*np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1)
	C[0, -1] = -1.0
	C[-1, 0] = -1.0
	return C


def periodicPoissonEigen(N):
	"""
	Closed-form spectrum of the circulant periodic Poisson matrix
	==============================================================
	    lambda_k = 4 sin^2(pi k / N),   k = 0, 1, ..., N-1,
	with the discrete Fourier modes as eigenvectors. Hence the bare QFT
	diagonalizes the operator directly (no sine transform, no odd extension).
	The null mode is lambda_0 = 0.
	"""
	k = np.arange(N)
	return 4.0*np.sin(np.pi*k/N)**2


def spectralPoissonCircuitPeriodic(f, measure=True):
	"""
	Periodic spectral-Poisson compliance circuit (native QFT)
	==========================================================
	On m = log2(N) system qubits plus one ancilla:
	  (1) encode the normalized load |psi_b> ,
	  (2) apply the bare QFT (the eigenbasis of the circulant operator),
	  (3) invert eigenvalues with a multiplexed Ry; the null mode k=0 is
	      assigned a zero angle, so it is skipped rather than dividing by zero.
	Because lambda_k = 4 sin^2(pi k/N) is closed form, no phase estimation is
	needed. The angle array is symmetric (lambda_k = lambda_{N-k}), so the
	result is insensitive to the QFT's internal qubit ordering.

	Returns (circuit, lam_min) where lam_min is the smallest *nonzero* eigenvalue.
	"""
	N = len(f)
	m = int(np.log2(N))
	if 2**m != N:
		raise ValueError("len(f) must be a power of 2")

	lam = periodicPoissonEigen(N)
	nonzero = lam > 1e-12
	lam_min = lam[nonzero].min()

	angles = np.zeros(N)                       # null mode (k=0) gets angle 0
	angles[nonzero] = 2.0*np.arcsin(np.sqrt(lam_min/lam[nonzero]))

	q = QuantumRegister(m, 'q')
	a = QuantumRegister(1, 'a')
	circuit = QuantumCircuit(q, a)

	# (1) encode normalized load (a self-equilibrated load has zero k=0 amplitude)
	psi = f/np.linalg.norm(f)
	circuit.prepare_state(Statevector(psi), list(q))

	# (2) bare QFT -- diagonalizes the circulant operator directly
	circuit.append(QFTGate(num_qubits=m), list(q))

	# (3) eigenvalue inversion, null mode skipped
	circuit.append(UCRYGate(list(angles)), [a[0]] + list(q))

	if measure:
		c = ClassicalRegister(1, 'c')
		circuit.add_register(c)
		circuit.measure(a[0], c[0])
	return circuit, lam_min


def complianceFromProbabilityPeriodic(f, P, lam_min):
	"""
	Rescale the ancilla probability into the periodic compliance
	=============================================================
	c = h^3 (f.f) P / lam_min  with the periodic grid spacing h = 1/N.
	The k=0 component is excluded (pseudoinverse), consistent with the
	zero-mean gauge and the self-equilibrated load.
	"""
	N = len(f)
	h = 1.0/N
	return h**3*(f @ f)*P/lam_min


def spectralComplianceQuantumPeriodic(f, shots=100000):
	"""Periodic compliance via the quantum pipeline (sampling stands in for QAE)."""
	circuit, lam_min = spectralPoissonCircuitPeriodic(f, measure=True)
	counts = simulate_measurements(circuit, shots=shots)
	P = counts.get('1', 0)/shots
	return complianceFromProbabilityPeriodic(f, P, lam_min), P


def spectralComplianceExactPeriodic(f):
	"""Statevector-exact periodic compliance through the same circuit."""
	circuit, lam_min = spectralPoissonCircuitPeriodic(f, measure=False)
	m = int(np.log2(len(f)))
	P = float(Statevector(circuit).probabilities([m])[1])
	return complianceFromProbabilityPeriodic(f, P, lam_min), P


def spectralComplianceClassicalPeriodic(f):
	"""Reference periodic compliance c = h^3 f^T C^+ f (pseudoinverse)."""
	N = len(f)
	h = 1.0/N
	return h**3*(f @ np.linalg.pinv(periodicPoissonStiffness(N)) @ f)
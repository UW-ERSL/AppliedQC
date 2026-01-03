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
from Chapter07_QuantumGates_functions import simulateCircuit

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
	return omega ** (i * j)

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
	counts = simulateCircuit(circuit,shots=shots)
		
	return counts

def processQFTResult(M,counts,shots):
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
	M : int
		Signal length (number of frequency bins)
	counts : dict
		Measurement results from QFT circuit
	shots : int
		Total number of measurements
		
	Returns:
	--------
	ampl : ndarray (M/2,)
		Amplitude spectrum (positive frequencies only due to symmetry)
		
	Note: Phase information is lost in measurement; only magnitudes recovered
	"""
	phi = np.zeros(M)
	# Extract amplitudes from measurement counts
	for i in counts:
		freq = int(i, 2)  # Convert bitstring to frequency index
		phi[freq] = np.sqrt(counts[i]/shots)  # Amplitude proportional to sqrt(probability)
	
	# Use symmetry to combine positive and negative frequencies
	ampl = (phi[1:int(M/2)])+(phi[M-1:int(M/2):-1])
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
	Total: O(n²) gates where n = log₂(M)
	
	Parameters:
	-----------
	m : int
		Number of qubits
		
	Returns:
	--------
	circuit : QuantumCircuit
		QFT circuit with m qubits
		
	Efficiency Note:
	This O(n²) gate complexity compares favorably to classical FFT's O(M log M)
	when expressed in terms of problem size M = 2^n: O(log²M) vs O(M log M)
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

def createQFTMatrix(M):
	"""
	Generate QFT transformation matrix
	===================================
	Creates unitary matrix representation of QFT.
	QFT[j,k] = (1/sqrt(M)) · ω^(jk) where ω = exp(2πi/M)
	
	Difference from DFT:
	- QFT uses positive sign in exponent: exp(+2πi/M)
	- QFT includes 1/sqrt(M) normalization (unitary)
	- DFT uses exp(-2πi/M) without normalization
	
	Parameters:
	-----------
	M : int
		Transform size (must be power of 2 for quantum implementation)
		
	Returns:
	--------
	QFTMatrix : ndarray (M × M)
		Complex unitary QFT matrix
		
	Properties:
	- Unitary: QFT† · QFT = I
	- Inverse: QFT^(-1) = QFT†
	- Eigenvalues all have magnitude 1
	"""
	omega = np.exp(1j*(2*np.pi/M))
	i, j = np.meshgrid(np.arange(M), np.arange(M))
	QFTMatrix = omega ** (i * j)/np.sqrt(M)  # Include normalization
	return QFTMatrix
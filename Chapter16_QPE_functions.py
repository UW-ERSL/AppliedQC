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
from qiskit.circuit.library import QFTGate
from Chapter08_QuantumGates_functions import simulateCircuit
from qiskit.circuit.library import QFT, phase_estimation, HamiltonianGate

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
	counts = simulateCircuit(circuit,shots=nShots)
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
	circuit.draw('mpl')
	
	# Execute circuit and process results
	counts = simulateCircuit(circuit,shots = nShots)
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
	print(n,m)
	
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
	
	# Measure with reversed classical bit order (Qiskit convention)
	circuit.measure( [*range(0, m)],[*range(m-1,-1,-1)])
	counts = simulateCircuit(circuit,shots = nShots,method ='statevector')
	
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
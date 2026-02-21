"""
HHL Algorithm - Quantum Algorithm for Solving Linear Systems
=============================================================
Harrow-Hassidim-Lloyd (2009) algorithm for solving Ax = b.

Problem: Given N×N Hermitian matrix A and vector b, find x such that Ax = b.

Quantum Advantage Conditions:
1. A is sparse (≤s non-zero entries per row) and efficiently row-accessible
2. Condition number κ = λ_max/λ_min is not too large
3. Only need to sample from solution (not full vector recovery)
4. b can be efficiently prepared as quantum state

Complexity Comparison:
- Classical (direct): O(N³) for dense matrices, O(Nκ log(1/ε)) for special cases
- Classical (iterative, conjugate gradient): O(Nκ log(1/ε)) for well-conditioned SPD
- HHL (quantum): O(log(N)s²κ²/ε) - exponential speedup in N

Algorithm Overview:
===================
Input: Hermitian matrix A, normalized state |b⟩, precision parameters
Output: Quantum state |x⟩ ∝ A^(-1)|b⟩

Steps:
1. [QPE] Estimate eigenvalues λ_j of A
   - Decompose |b⟩ = Σ_j β_j|u_j⟩ where A|u_j⟩ = λ_j|u_j⟩
   - QPE creates: Σ_j β_j|λ_j⟩|u_j⟩
   
2. [Rotation] Conditional rotation to encode λ^(-1)
   - For each eigenvalue λ_j: rotate ancilla by θ_j ∝ λ_j^(-1)
   - Creates: Σ_j β_j|λ_j⟩(√(1-C²/λ_j²)|0⟩ + C/λ_j|1⟩)|u_j⟩
   - C scales the inversion (C = λ_min for proper normalization)
   
3. [Uncompute] Reverse QPE (apply inverse QPE)
   - Disentangle eigenvalue register
   
4. [Measure] Measure ancilla
   - Success (ancilla = 1): solution state ∝ Σ_j (β_j/λ_j)|u_j⟩ = |x⟩
   - Failure (ancilla = 0): retry (success probability ∝ 1/κ²)

Key Parameters:
---------------
- m: Number of ancilla qubits for QPE precision (Δλ ~ 1/2^m)
- λ_upper: Upper bound on |λ|
- λ_lower: Lower bound on |λ| (typically estimated from data)
- P0: Probability cutoff for filtering QPE results

Success Probability:
-------------------
P_success ∝ (λ_min/λ_max)² = 1/κ²
For poorly conditioned systems, may need O(κ²) repetitions.

Applications in Optimization:
-----------------------------
- Solving KKT systems in constrained optimization
- Newton step computation: Hx = -∇f (Hessian system)
- Topology optimization: iterative FEM solves
- PDE-constrained optimization
- Note: HHL provides ||x|| efficiently, but extracting all x_i requires O(N) measurements

Implementation Notes:
--------------------
This hybrid version:
1. Simulates QPE circuit separately to extract eigenvalue estimates
2. Uses estimates to construct controlled rotations
3. Constructs and simulates full HHL circuit
4. Post-processes to extract solution state

Limitations:
-----------
- Requires A to be Hermitian (use A^TA for general matrices)
- Full vector recovery requires O(N) measurements (no speedup)
- Practical advantage for: sampling, computing norms, inner products
- Current hardware: limited qubits, noise affects accuracy

References:
-----------
- Harrow, Hassidim, Lloyd (2009): "Quantum algorithm for linear systems of equations" 
  Physical Review Letters 103(15)
- Childs et al. (2017): "Quantum algorithm for systems of linear equations with 
  exponentially improved dependence on precision"
- Duan et al. (2020): "Survey on HHL algorithm: From theory to application in quantum  machine learning"
"""

import scipy
import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT, HamiltonianGate
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit_aer import Aer
from qiskit.circuit.library import QFTGate
from Chapter08_QuantumGates_functions import simulateCircuit
class myHHL:
	"""
	Hybrid HHL Algorithm Implementation
	====================================
	Combines quantum circuits for QPE and controlled rotations with classical
	post-processing to solve Ax = b.
	
	Architecture:
	1. QPE Circuit: Separately estimates eigenvalues of A
	2. HHL Circuit: Uses QPE estimates to perform controlled rotations
	3. Post-processing: Extracts solution from measurement results
	
	This "hybrid" approach allows inspection of intermediate QPE results
	and adaptive construction of rotation angles based on estimated eigenvalues.
	
	Key Innovation:
	Instead of hardcoding rotation angles, this implementation:
	- Runs QPE first to identify dominant eigenvalues
	- Prunes low-probability eigenvalues (below P0 threshold)
	- Computes optimal rotation angles for identified eigenvalues
	- Constructs HHL circuit with these targeted rotations
	
	Attributes (after initialization):
	----------------------------------
	A : ndarray (N×N)
		System matrix (must be Hermitian)
	b : ndarray (N,)
		RHS vector (must be normalized)
	lambdaUpper : float
		Upper bound on |λ_max|
	lambdaLower : float
		Lower bound on |λ_min| (estimated during execution)
	m : int
		QPE precision (number of phase qubits)
	n : int
		log₂(N) - qubits for encoding vectors
	"""
	def __init__(self, A, b, lambdaUpper,
			  m = 3,P0 = 0.1,nShots = 1000):
		"""
		Initialize HHL solver with problem data and parameters
		
		Parameters:
		-----------
		A : ndarray (N×N)
			System matrix, must be:
			- Hermitian (A = A†)
			- Power-of-2 dimension for quantum encoding
			- Well-conditioned (κ = λ_max/λ_min not too large)
		b : ndarray (N,)
			Right-hand side, must be:
			- Unit normalized (||b|| = 1)
			- Compatible with A
		lambdaUpper : float
			Upper bound on eigenvalue magnitudes |λ| ≤ lambdaUpper
			Used for Hamiltonian time evolution: t = -2π/λ_upper
			Recommendation: Use 1.1 × max(Gershgorin bounds) for safety
		m : int (default 3)
			QPE precision bits. Phase resolution = 1/2^m
			Trade-off: Higher m → better accuracy but deeper circuits
			Typical: m=3-5 for small problems, m=8-10 for production
		P0 : float (default 0.1)
			Probability cutoff for eigenvalue filtering
			QPE results with P < P0 are discarded
			Reduces noise from spurious eigenvalue estimates
		nShots : int (default 1000)
			Number of circuit shots for both QPE and HHL
			More shots → better statistics, longer runtime
			
		Validation Checks:
		-----------------
		- N is power of 2 (required for quantum encoding)
		- A is symmetric (Hermitian for real matrices)
		- b is normalized
		- Compatible dimensions
		- Reasonable problem size (n ≤ 10 for simulation)
		"""	
		self.A = A
		self.b = b
		self.lambdaUpper = lambdaUpper  # Used in Hamiltonian evolution
		self.m = m  # Number of bits to estimate eigenvalues
		self.nHHLShots = nShots  # Shots for simulating HHL circuit
		self.nQPEShots = nShots  # Shots for simulating QPE circuit
		self.N = self.A.shape[0]
		self.n = int(np.log2(self.N))  # Qubits needed for b and solution
		self.dataOK = True
		self.probabilityCutoff = P0  # For pruning QPE eigenphases
		
		# Validation: Check N is power of 2
		if np.abs(2**self.n - self.A.shape[0]) > 1e-10: 
			print("Invalid size of matrix; must be power of 2") 
			self.dataOK = False
			
		# Practical limit for classical simulation
		if (self.n > 10):
			print('Matrix size is too large for classical simulation')
			print(f'Current: 2^{self.n} = {self.N}, Maximum recommended: 2^10 = 1024')
			self.dataOK = False
			
		# Validation: Check A is symmetric (Hermitian for complex)
		symErr = np.max(np.abs(self.A - np.transpose(self.A)))
		if (symErr > 1e-10):
			print(f'A does not appear to be symmetric (max error: {symErr})')
			self.dataOK = False
			
		# Validation: Check b is normalized
		if (not np.isclose(np.linalg.norm(b), 1.0)):
			print(f'b does not appear to be of unit magnitude (||b|| = {np.linalg.norm(b)})')
			self.dataOK = False
			
		# Validation: Check m is reasonable
		if (m < 1):
			print('m has to be at least 1 (but 3+ recommended for meaningful precision)')
			self.dataOK = False
			
		# Validation: Check dimensional compatibility
		if not (A.shape[0] == b.shape[0]):
			print(f'A and b sizes are not compatible ({A.shape[0]} vs {b.shape[0]})')
			self.dataOK = False
			
		self.debug = False
	
	def solveuExact(self):
		"""
		Classical solver for verification (not part of HHL algorithm)
		Solves Ax = b classically, normalizes to |u⟩ = x/||x|| for comparison
		"""
		self.xExact = scipy.linalg.solve(self.A, self.b)
		self.uExact = self.xExact/np.linalg.norm(self.xExact)

	def computeEigen(self):
		"""
		Classical eigendecomposition for verification and analysis
		Extracts eigenvalues and eigenvectors of A
		Note: Assumes A is symmetric, so eigenvalues are real
		"""
		self.eig_val, self.eig_vec = scipy.linalg.eig(self.A)
		# Remove imaginary component (should be ~0 for symmetric A)
		self.eig_val = np.abs(self.eig_val)

	def estimate_lambda_bounds_gershgorin(self):
		"""
		Estimate eigenvalue bounds using Gershgorin Circle Theorem
		==========================================================
		
		Theorem: For matrix A, eigenvalues lie within union of discs:
		|λ - A[i,i]| ≤ Σ_{j≠i} |A[i,j]| for each row i
		
		Provides guaranteed bounds without computing eigenvalues explicitly.
		Critical for HHL: λ_lower determines rotation angle normalization.
		
		Returns:
		--------
		(λ_lower, λ_upper) : tuple of floats
			Conservative bounds on eigenvalue range
			
		Application: Sets C = λ_lower in controlled rotation:
		θ = 2·arcsin(C/λ) ensures valid angles and proper normalization
		"""
		n = self.A.shape[0]
		lambda_upper_bound = -np.inf
		lambda_lower_bound = np.inf
		
		# Row-wise Gershgorin circles
		for i in range(n):
			center = self.A[i, i]  # Diagonal element
			radius = np.sum(np.abs(self.A[i, :])) - np.abs(self.A[i, i])  # Off-diagonal sum
			lambda_upper_bound = max(lambda_upper_bound, center + radius)
			lambda_lower_bound = min(lambda_lower_bound, center - radius)
		
		# For SPD matrices, ensure lower bound > 0
		if lambda_lower_bound <= 0:
			# Try column-wise Gershgorin if row-wise gives non-positive bound
			for j in range(n):
				center = self.A[j, j]
				radius = np.sum(np.abs(self.A[:, j])) - np.abs(self.A[j, j])
				lambda_lower_bound = max(lambda_lower_bound, center - radius)
		
		return lambda_lower_bound, lambda_upper_bound

	

	def constructQPECircuit(self,mode):
		"""
		Construct QPE circuit for HHL - three modes:
		=============================================
		
		Modes:
		------
		'QPE': Standalone QPE for eigenvalue estimation
			   Returns eigenvalues of A when given eigenstate
			   
		'HHLFront': First half of HHL (QPE + eigenvalue encoding)
					Prepares state with eigenvalue information entangled
					
		'HHLRear': Second half of HHL (used for uncomputing QPE)
				   Inverse of HHLFront to disentangle eigenvalue register
		
		Circuit Structure:
		- phase_qubits (m): Ancilla for QPE, stores binary phase estimate
		- b_qubits (n): Data register, holds input state |b⟩ and solution
		- ancilla_qubit (1): Success indicator for HHL (modes 'HHLFront'/'HHLRear' only)
		
		QPE Process:
		1. Initialize phase qubits to |+⟩^⊗m
		2. Prepare |b⟩ in data register
		3. Apply controlled-U^(2^k) for k=0..m-1 where U = exp(iAt)
		4. Apply inverse QFT to phase register → binary eigenvalue estimate
		"""
		phase_qubits = QuantumRegister(self.m, 'θ')
		b_qubits = QuantumRegister(self.n, 'b')
		
		if (mode == 'QPE'):
			offset = 0  # No ancilla qubit
			cl = ClassicalRegister(self.m,'cl')
			circuit = QuantumCircuit( phase_qubits, b_qubits,cl)
		elif (mode == 'HHLFront') or (mode == 'HHLRear'):
			ancilla_qubit = QuantumRegister(1,'a')  # For controlled rotation success
			offset = 1  # Shift indices due to ancilla
			cl = ClassicalRegister(1+self.n,'cl')
			circuit = QuantumCircuit(ancilla_qubit, phase_qubits, b_qubits,cl)	
		else:
			print('Incorrect mode in constructQPECircuit')
			return
			
		# Step 1: Hadamards on phase qubits → uniform superposition
		for i in range(self.m):
			circuit.h(offset+i)
			
		# Step 2: Prepare input state (for QPE or HHLFront only)
		if (mode == 'QPE') or (mode == 'HHLFront'):
			circuit.prepare_state(Statevector(self.b),
						 [*range(self.m+offset, self.n+self.m+offset)],'b')
			
		# Step 3: Hamiltonian evolution with normalized time
		# t = -2π/λ_upper maps λ_upper → phase 2π → θ = 1
		t = -2*np.pi/self.lambdaUpper
		U_A = HamiltonianGate(self.A, time=t,label = 'UA')
		U_A._name = 'UA'
		
		# Step 4: Controlled-U^(2^k) operations for phase kickback
		for i in range(self.m):
			U_A_pow = U_A.power(2**i)  # Repeated squaring: U, U², U⁴, ...
			UControl = U_A_pow.control(1)
			circuit.append(UControl,[i+offset,*range(self.m+offset, self.n+self.m+offset)])
		
		# Step 5: Inverse QFT extracts phase from superposition
		qft = QFTGate(num_qubits=self.m).inverse()
		if (mode == 'QPE') or (mode == 'HHLFront'):
			qft._name = 'IQFT'
		elif (mode == 'HHLRear'):
			qft._name = 'QFT'  # Will be inverted again when this circuit is inverted
		circuit.append(qft, [*range(offset,self.m+offset)])
		return circuit
	
	def processQPECounts(self,counts):
		"""
		Process QPE measurement results to extract eigenvalue estimates
		================================================================
		
		Filters and sorts eigenvalue estimates based on:
		1. Measurement frequency (probability)
		2. Validity (non-zero eigenvalues)
		3. Probability threshold P0
		
		Creates self.thetaTilde: array of normalized phases θ = λ/λ_upper
		Only includes phases with P > P0 and θ ≠ 0
		"""
		# Sort by measurement count (descending)
		self.QPECountsSorted = {k: v for k, v in sorted(self.QPECounts.items(), 
											 key=lambda item: item[1],
											 reverse=True)}
		self.thetaTilde  = np.array([])	
		for key in self.QPECountsSorted:
			thetaValue = int(key, 2)/(2**self.m)  # Binary to decimal, normalize
			probability = self.QPECountsSorted[key]/self.nQPEShots
			
			# Filter: skip zero eigenvalues (singular matrix) and low-probability estimates
			if ((thetaValue == 0) or (probability < self.probabilityCutoff)):
				continue
			self.thetaTilde  = np.append(self.thetaTilde,thetaValue)
	
	def constructHHLCircuit(self):
		"""
		Construct full HHL circuit using QPE eigenvalue estimates
		==========================================================
		
		Circuit Structure:
		1. QPE Front: Prepare superposition with eigenvalue information
		2. Controlled Rotation: For each estimated eigenvalue θ_j:
		   Rotate ancilla by α = 2·arcsin(λ_lower/λ_j)
		   Success: ancilla = |1⟩ with amplitude ∝ λ_lower/λ_j
		3. Inverse QPE: Disentangle phase register
		4. Measure: ancilla (success?) + data qubits (solution state)
		
		Key Innovation - Adaptive Rotations:
		Instead of uniform rotations, uses actual QPE estimates to:
		- Skip spurious low-probability eigenvalues
		- Skip eigenvalues below λ_lower (would give invalid rotations)
		- Optimize rotation angles for detected eigenvalues
		
		Result: |ψ⟩ = (1/Z) Σ_j (β_j/λ_j)|u_j⟩|1⟩ + (unwanted)|0⟩
		where Z is normalization and success when measuring |1⟩ on ancilla
		"""
		self.HHLCircuit = self.constructQPECircuit('HHLFront')
		self.HHLCircuit.barrier()
		
		# Controlled rotation for each estimated eigenvalue
		for key in self.QPECountsSorted:
			thetaValue = int(key, 2)/(2**self.m)  # Normalized phase
			probability = self.QPECountsSorted[key]/self.nQPEShots
			
			# Filter low-probability and zero eigenvalues
			if (thetaValue == 0) or (probability < self.probabilityCutoff):
				continue
				
			# Convert normalized phase back to eigenvalue estimate
			lambdaTilde = thetaValue*self.lambdaUpper
			
			# Skip if eigenvalue below lower bound (invalid for rotation)
			if (self.lambdaLower > lambdaTilde):
				continue
			
			# Rotation angle: α = 2·arcsin(C/λ) where C = λ_lower
			# This maps λ → ancilla rotation such that:
			# |0⟩ → cos(α/2)|0⟩ + sin(α/2)|1⟩
			# Amplitude in |1⟩ ∝ sin(arcsin(C/λ)) = C/λ ∝ 1/λ (desired!)
			alpha = 2*np.arcsin(self.lambdaLower/lambdaTilde)	
			
			# Multi-controlled rotation: only apply when phase qubits match 'key'
			cu_gate = UGate(alpha, 0, 0).control(self.m, ctrl_state = key) 
			self.HHLCircuit.append(cu_gate,[*range(1,1+self.m),0])  # Control: phases, Target: ancilla
			
		# Uncompute QPE (disentangle eigenvalue information)
		qpeInverse = self.constructQPECircuit('HHLRear').inverse()
		self.HHLCircuit.barrier()
		self.HHLCircuit.compose(qpeInverse,
						  [*range(0,1+self.m+self.n)], 
						  [*range(0,1+self.n)],inplace = True)
		
		self.HHLCircuit.barrier()
		# Measure ancilla (success indicator) and data qubits (solution)
		self.HHLCircuit.measure(qubit = [0,*range(1+self.m,1+self.m+self.n)], 
							  cbit =  [0,*range(1,self.n+1)])
	
	def extractHHLSolution(self):
		"""
		Extract solution state from HHL measurement results
		===================================================
		
		Post-selection: Keep only shots where ancilla = 1 (success)
		For these successful shots, the data qubits encode |x⟩ ∝ A^(-1)|b⟩
		
		Process:
		1. Filter measurements: keep only ancilla = 1 outcomes
		2. For each successful measurement bitstring:
		   - Convert to statevector |v⟩
		   - Weight by √(measurement frequency)
		3. Sum weighted states → approximation to |x⟩
		
		Success Rate:
		P_success ∝ (λ_min/λ_max)² = 1/κ²
		For ill-conditioned systems, may have few successful shots
		
		Returns: True if successful shots found, False otherwise
		"""
		self.HHLSuccessCounts = {}
		nSuccessCounts = 0
		
		# Filter for successful outcomes (ancilla = 1)
		for key in self.HHLRawCounts:
			if (key[-1] == '1'):  # Last bit is ancilla
				self.HHLSuccessCounts[key] = self.HHLRawCounts[key]
				nSuccessCounts += self.HHLSuccessCounts[key]
				
		self.uHHL = 0
		if (nSuccessCounts == 0):
			print("Warning: No successful HHL measurements (ancilla=1)")
			print("Try: more shots, better λ_lower estimate, or check condition number")
			return False
			
		# Reconstruct solution state from successful measurements
		for key in self.HHLSuccessCounts:
			subkey = key[0:-1]  # Remove ancilla bit
			v = np.real(Statevector.from_label(subkey))  # Bitstring → statevector
			# Weight by square root of frequency (amplitude, not probability)
			self.uHHL = self.uHHL + v*(np.sqrt(self.HHLSuccessCounts[key]/nSuccessCounts))
		return True
	
	def executeHHL(self):
		"""
		Main HHL execution pipeline
		============================
		
		Complete workflow:
		1. Validate input data
		2. Estimate λ_lower using Gershgorin + QPE
		3. Run QPE to extract eigenvalue spectrum
		4. Construct HHL circuit with adaptive rotations
		5. Simulate HHL circuit
		6. Post-process to extract solution state
		
		Returns: True if solution extracted successfully, False otherwise
		
		Usage:
		------
		>>> hhl = myHHL(A, b, lambdaUpper=10, m=4)
		>>> if hhl.executeHHL():
		>>>     print("Solution:", hhl.uHHL)
		>>>     print("Classical:", hhl.uExact)  # after calling solveuExact()
		"""
		if not self.dataOK:
			print('Check input data - validation failed in __init__')
			return False
			
		# Estimate eigenvalue bounds using Gershgorin theorem
		lambda_lower_gershgorin, lambda_upper_gershgorin = self.estimate_lambda_bounds_gershgorin()

		## Step 1: Run QPE to estimate eigenvalues of A
		self.QPECircuit = self.constructQPECircuit('QPE')
		self.QPECircuit.measure([*range(0,self.m)], [*range(0,self.m)]) 
		self.QPECounts = simulateCircuit(self.QPECircuit,shots = self.nQPEShots)
		self.processQPECounts(self.QPECounts)
		
		# Step 2: Refine λ_lower estimate using QPE results
		# Use QPE-based estimate if tighter than Gershgorin
		if len(self.thetaTilde) > 0:
			lambda_lower_qpe = 0.99 * min(self.thetaTilde) * self.lambdaUpper  # 0.99 for safety margin
			self.lambdaLower = max(lambda_lower_gershgorin * 0.99, lambda_lower_qpe)
		else:
			self.lambdaLower = lambda_lower_gershgorin * 0.99
			print("Warning: No eigenvalues detected by QPE, using Gershgorin only")
			
		print(f"λ_lower (Gershgorin): {lambda_lower_gershgorin:.6f}")
		if len(self.thetaTilde) > 0:
			print(f"λ_lower (QPE): {lambda_lower_qpe:.6f}")
		print(f"λ_lower (used): {self.lambdaLower:.6f}")

		# Step 3: Construct and run full HHL circuit
		self.constructHHLCircuit()
		self.HHLRawCounts = simulateCircuit(self.HHLCircuit, shots=self.nHHLShots)
		
		# Step 4: Extract solution from post-selected measurements
		if not self.extractHHLSolution():
			return False
		return True
	
	def debugHHL(self):
		"""
		Debugging and verification utility
		===================================
		Prints detailed diagnostics:
		- Classical eigenvalues and eigenvectors
		- Expected eigenphases for QPE
		- Classical solution for comparison
		- Validates λ_upper choice
		
		Use before executeHHL() to verify problem setup
		"""
		self.computeEigen()
		print("Exact eigen values of A:\n", self.eig_val)
		print("Exact eigen vectors of A:\n", self.eig_vec)
		eigenPhase = self.b*self.eig_val/self.lambdaUpper
		tMax = 2*np.pi/max(self.eig_val)
		print("Exact eigenphases of A:\n", eigenPhase)
		if (max(eigenPhase) >= 1):
			print('Invalid value for lambdaUpper and/or fs', self.lambdaUpper,self.f)
			print('Here choose t to be <', abs(tMax))
		
		self.solveuExact()
		print('xExact:', self.xExact)
		
if __name__ == "__main__":
	example = 1
	debug = False
	nShots = 1000
	if (example == 1):
		A = np.array([[1,0],[0,0.75]])
		v0 = np.array([0,1])
		v1 = np.array([1,0])
		D = [0.75,1]
		b = np.array([1/np.sqrt(2),1/np.sqrt(2)])
		lambdaUpper = 2
		m = 2
		P0 = 0.1
	elif (example == 2):
		A = np.array([[2,-1],[-1,2]])
		v0 = np.array([1/np.sqrt(2),1/np.sqrt(2)])
		v1 = np.array([1/np.sqrt(2),-1/np.sqrt(2)])
		D = [1,3]
		b = np.array([-1/np.sqrt(2),1/np.sqrt(2)])
		lambdaUpper = 6
		m = 5
		P0 = 0.1
	elif (example == 3):
		A = np.array([[1,0,0,-0.5],[0,1,0,0],[0,0,1,0],[-0.5,0,0,1]])
		v0 = np.array([1/np.sqrt(2),0,0,1/np.sqrt(2)])
		v1 = np.array([0,1,0,0])
		v2 = np.array([0,0,1,0])
		v3 = np.array([1/np.sqrt(2),0,0,-1/np.sqrt(2)])
		D = [0.5,1,1,1.5]
		#a = [1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4),1/np.sqrt(4)]
		a = [0,0,1,0]
		b = a[0]*v0 + a[1]*v1 + a[2]*v2 + a[3]*v3
		lambdaUpper = 3
		m = 2
		P0 = 0.1
	elif (example == 4):
		N = 8 # Order of matrix
		# (tridiagonal: 4 on diagonal, -1 on off-diagonals)
		A = np.zeros((N, N))
		for i in range(N):
			A[i, i] = 4
			if i > 0:
				A[i, i-1] = -1
			if i < N-1:
				A[i, i+1] = -1
		
		# Create a simple normalized vector b
		b = np.ones(N) / np.sqrt(N)
		
		lambdaUpper = 6 # Upper bound for eigenvalues (max eigenvalue is ~4 for finite difference)
		m = 3
		P0 = 0.1


	print('A:\n', A)
	print('b:\n', b)
	HHL = myHHL(A,b,lambdaUpper = lambdaUpper,
				 m = m,P0 = P0, nShots = nShots)
	
	# Execute main code
	if debug:
		HHL.debugHHL()
	if (HHL.executeHHL()):
		print("uHHL: \t\t\t", HHL.uHHL)
		HHL.solveuExact()
		print('uExact: \t\t', HHL.uExact)
		fidelity = np.dot(HHL.uHHL,HHL.uExact)
		print('fidelity:', fidelity)
		
"""
Variational Quantum Linear Solver (VQLS)
=========================================
Alternative quantum algorithm for solving Ax = b using variational optimization.

Reference: Bravo-Prieto, C. et al. (2023). "Variational quantum linear solver." 
           Quantum, 7, p.1188.

Key Difference from HHL:
------------------------
- HHL: Uses QPE (requires deep circuits, many ancilla qubits)
- VQLS: Variational approach (shallower circuits, suitable for NISQ devices)

Algorithm Overview:
===================
Goal: Find quantum state |x⟩ such that A|x⟩ ∝ |b⟩

Approach: Parametrized Quantum Circuit (Ansatz)
1. Prepare trial state: |x(θ)⟩ = U(θ)|0⟩
2. Define cost function: C_G = 1 - |⟨b|A|x(θ)⟩|² / (⟨x(θ)|A†A|x(θ)⟩)
3. Minimize C_G classically (e.g., COBYLA optimizer)
4. Optimal θ* gives |x*⟩ ≈ A^(-1)|b⟩

Cost Function Components:
-------------------------
Numerator: |⟨b|A|x(θ)⟩|² = |Σ_j α_j⟨b|P_j|x(θ)⟩|²
   where A = Σ_j α_j P_j (Pauli decomposition)
   Measured via Hadamard test circuits

Denominator: ⟨x(θ)|A†A|x(θ)⟩ = Σ_{j,k} α_j*α_k⟨x(θ)|P_j†P_k|x(θ)⟩
   Also measured via Hadamard tests

Advantages of VQLS:
-------------------
+ Shallow circuits: Better suited for near-term quantum devices
+ No QPE needed: Avoids controlled-U^(2^k) operations
+ Flexible ansatz: Can incorporate problem structure
+ Robust to noise: Variational optimization handles some errors

Disadvantages:
--------------
- Classical optimization overhead: Many circuit evaluations
- No exponential speedup guarantee: Depends on cost landscape
- Barren plateaus: May have flat regions in parameter space
- Measurement cost: O(n_Pauli² × shots) per iteration

Complexity Comparison:
----------------------
- HHL: O(log(N)κ²/ε) queries, deep circuits
- VQLS: Shallow circuits, but O(iterations × n_Pauli² × shots)
- Classical: O(N³) direct, O(Nκ) iterative

When to Use VQLS vs HHL:
------------------------
Use VQLS when:
- Limited qubit coherence (NISQ devices)
- Shallow circuits needed
- No QPE available
- Willing to trade time for circuit depth

Use HHL when:
- High-quality qubits available
- Deep circuits feasible
- Need provable speedup
- Problem is well-conditioned

Applications in Optimization:
-----------------------------
- Iterative linear solves in topology optimization
- Newton steps: minimize ||Hx + ∇f||
- Constraint satisfaction: solve KKT systems
- Time-dependent problems: evolve with shallow circuits
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT, HamiltonianGate
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit_aer import Aer
from qiskit.circuit.library import QFTGate
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import UnitaryGate, QFTGate
from qiskit.circuit.library import QFT, phase_estimation, HamiltonianGate
from Chapter08_QuantumGates_functions import simulateCircuit
from qiskit.quantum_info import SparsePauliOp
class myVQLS:
	"""
	Variational Quantum Linear Solver Implementation
	=================================================
	Solves Ax = b by minimizing cost function C_G using parametrized quantum circuits.
	
	Algorithm Components:
	1. Ansatz: Parametrized circuit U(θ) generating trial states
	2. Hadamard Test: Quantum circuits to measure ⟨b|A|x(θ)⟩ and ⟨x(θ)|A†A|x(θ)⟩
	3. Cost Function: C_G = 1 - |numerator|²/denominator
	4. Classical Optimizer: Adjusts θ to minimize C_G
	
	Key Innovation:
	Instead of requiring QPE, uses shallow variational circuits that can run on NISQ hardware.
	Trade-off: No exponential speedup guarantee, but practical for near-term devices.
	"""
	def __init__(self, A, b,nShots = 10000):
		"""
		Initialize VQLS solver
		
		Parameters:
		-----------
		A : ndarray (N×N)
			System matrix (does not need to be Hermitian for VQLS)
			Will be decomposed into Pauli operators: A = Σ α_j P_j
		b : ndarray (N,)
			Right-hand side, must be normalized ||b|| = 1
		nShots : int (default 10000)
			Number of shots per circuit evaluation
			Higher shots → better cost estimates but longer runtime
			Recommendation: 10000-100000 for accurate gradients
			
		Ansatz Structure:
		----------------
		- Initial layer: R_y(θ) on all qubits
		- Repeated layers: CZ entanglers + R_y rotations
		- Number of layers: min(n-1, 4) for efficiency
		- Total parameters: n + layers×(n + n-2)
		
		Pauli Decomposition:
		-------------------
		A is decomposed as A = Σ_j α_j P_j where P_j are Pauli operators
		Number of terms: O(4^n) worst case, but typically sparse
		"""	
		self.A = A
		self.b = b
		self.nShots = nShots
		self.N = self.A.shape[0]
		self.n = int(np.log2(self.N))  # Number of qubits
		self.CostIterations = []  # Track optimization progress
		
		# Determine ansatz depth: balance expressivity vs circuit depth
		if (self.n <=4):
			self.ansatzLayers = self.n-1  # Full entanglement for small systems
		else:
			self.ansatzLayers = 4  # Cap depth for larger systems (NISQ limitation)
			
		self.dataOK = True
		
		# Calculate total number of variational parameters
		# Initial layer: n parameters
		# Each subsequent layer: (n parameters for R_y) + (n-2 for partial CZ)
		self.nAnsatzParams = self.n + self.ansatzLayers*(self.n + self.n-2)
		
		# Initialize parameters randomly
		self.ansatzParams = np.random.rand(self.nAnsatzParams)
		
		# Validation checks
		if np.abs(2**self.n - self.A.shape[0]) > 1e-10: 
			print("Invalid size of matrix; must be power of 2") 
			self.dataOK = False

		if (not np.isclose(np.linalg.norm(b), 1.0)):
			print(f'b does not appear to be of unit magnitude (||b|| = {np.linalg.norm(b)})')
			self.dataOK = False
			
		if not (A.shape[0] == b.shape[0]):
			print('A and b sizes are not compatible')
			self.dataOK = False
			
		self.debug = False
		
		# Precompute matrices needed for Hadamard tests
		self.computeUbMatrix()  # Unitary that prepares |b⟩
		self.PauliExpansion()   # Decompose A into Pauli basis
		
	def solveuExact(self):
		"""Classical solver for verification"""
		xExact = scipy.linalg.solve(self.A, self.b)
		self.uExact = xExact/np.linalg.norm(xExact)
		
	def computeUbMatrix(self):
		"""
		Compute unitary U_b that prepares state |b⟩ from |0⟩
		Used in Hadamard test circuits for measuring ⟨b|A|x(θ)⟩
		"""
		nQubits = self.n
		u =  QuantumRegister(nQubits, 'u')
		circuit = QuantumCircuit(u)
		circuit.prepare_state(Statevector(self.b) ,u)
		self.UbMatrix = Operator(circuit).data
		
	def PauliExpansion(self):
		"""
		Decompose matrix A into Pauli basis
		======================================
		A = Σ_j α_j P_j where P_j ∈ {I, X, Y, Z}^⊗n
		
		Enables measurement via Hadamard tests on each Pauli term.
		Number of terms: Up to 4^n, but typically much sparser.
		
		Cost Impact: O(n_terms²) circuit evaluations per cost function call
		"""
		pauliSplit = SparsePauliOp.from_operator(self.A)
		self.PauliMatrices = []
		for pauliOp in pauliSplit.paulis:
			PauliMatrix = pauliOp.to_matrix()
			self.PauliMatrices.append(PauliMatrix)	
		self.PauliCoefficients = pauliSplit.coeffs
		self.nPauliTerms = len(self.PauliCoefficients)
		return

	def createAnsatzCircuit(self,addMeasurement = False):
		"""
		Create parametrized ansatz circuit U(θ)
		========================================
		Hardware-efficient ansatz with alternating layers:
		1. R_y rotations on all qubits
		2. CZ entanglers (even pairs, then odd pairs)
		3. R_y rotations again
		
		Structure (per layer):
		- Even CZ: (0,1), (2,3), (4,5), ...
		- Odd CZ: (1,2), (3,4), (5,6), ...
		
		Expressivity: Can represent wide variety of states
		Depth: O(layers × n) gates, suitable for NISQ devices
		"""
		nQubits = self.n
		u =  QuantumRegister(nQubits, 'u')
		if (addMeasurement):
			c = ClassicalRegister(nQubits,'c')
			circuit = QuantumCircuit(u,c)
		else:
			circuit = QuantumCircuit(u)
			
		ansatzCounter = 0
		# Initial R_y layer
		for i in range(nQubits):
			circuit.ry(self.ansatzParams[ansatzCounter], u[i])
			ansatzCounter = ansatzCounter+1 
		circuit.barrier()
		
		# Repeated entangling + rotation layers
		for layer in range(self.ansatzLayers):
			# Even CZ pairs: (0,1), (2,3), ...
			for i in range(0,nQubits-1,2):
				circuit.cz(u[i], u[i+1])
			# R_y rotations on all qubits
			for i in range(nQubits):
				circuit.ry(self.ansatzParams[ansatzCounter], u[i])
				ansatzCounter = ansatzCounter+1 
			# Odd CZ pairs: (1,2), (3,4), ...
			for i in range(1,nQubits-1,2):
				circuit.cz(u[i], u[i+1])
			# R_y rotations on interior qubits
			for i in range(1,nQubits-1,1):
				circuit.ry(self.ansatzParams[ansatzCounter], u[i])
				ansatzCounter = ansatzCounter+1 	
			circuit.barrier()
		
		if (addMeasurement):
			circuit.measure(u, c)
		return circuit

	def HadamardTestCircuit1(self,UthetaMatrix, PjMatrix,UbMatrix,imagComponent = False):
		"""
		Hadamard Test for ⟨b|P_j|x(θ)⟩
		================================
		Measures real/imaginary part of matrix element using 1 ancilla qubit.
		
		Circuit: H -- Controlled-U -- H -- Measure
		where U = U_b† P_j U_θ
		
		Result: P(0) - P(1) gives Re[⟨b|P_j|x(θ)⟩] or Im[...] if S† gate added
		"""
		nQubits = self.n
		a = QuantumRegister(1, 'a')  # Ancilla for Hadamard test
		u =  QuantumRegister(nQubits, '0')
		c = ClassicalRegister(1,'c')
		circuit = QuantumCircuit(a,u,c)
		circuit.h(0)
		if (imagComponent):
			circuit.sdg(0)  # Phase shift for imaginary component

		# Combined unitary: U_b† P_j U_θ
		U = UnitaryGate(np.matmul(np.matmul(UbMatrix.conj().T,PjMatrix),UthetaMatrix))
		U._name = r'$ U_{\theta}$'
		UControl = U.control(1)
		circuit.append(UControl,[*range(self.n+1)])

		circuit.h(0)
		circuit.measure([0], [0])
		return circuit


	def HadamardTestCircuit2(self,UthetaMatrix, PjMatrix,PkMatrix,imagComponent = False):
		"""
		Hadamard Test for ⟨x(θ)|P_j† P_k|x(θ)⟩
		=========================================
		Measures matrix element between Pauli operators.
		Used for denominator of cost function.
		
		Circuit structure: 
		H -- U_θ -- Controlled-P_j† -- Controlled-P_k -- H -- Measure
		"""
		nQubits = self.n
		u =  QuantumRegister(nQubits, '0')
		a = QuantumRegister(1, 'a')
		c = ClassicalRegister(1,'c')
		circuit = QuantumCircuit(a,u,c)
		circuit.h(0)
		
		# Apply ansatz
		UTheta = UnitaryGate(UthetaMatrix)
		UTheta._name = r'$ U_{\theta}$'
		circuit.append(UTheta, [*range(1,self.n+1)])
		
		if (imagComponent):
			circuit.sdg(0)
		
		# Controlled Pauli operations
		UPj = UnitaryGate(PjMatrix).adjoint()
		UPj._name = r'$ U_{P_j}^{\dag}$'
		UPjControl = UPj.control(1)
		circuit.append(UPjControl, [*range(self.n+1)])
		
		UPk = UnitaryGate(PkMatrix)
		UPk._name = r'$ U_{P_k}$'
		UPkControl = UPk.control(1)
		circuit.append(UPkControl, [*range(self.n+1)])
		
		circuit.h(0)
		circuit.measure([0], [0])
		return circuit
		
	def costFunctionExact(self,thetaParams):
		"""
		Classical cost function evaluation (for testing/comparison)
		===========================================================
		Computes C_G = 1 - |⟨b|A|x(θ)⟩|² / ⟨x(θ)|A†A|x(θ)⟩ classically
		without Hadamard tests. Used for debugging and comparing with quantum version.
		"""
		self.ansatzParams = thetaParams.copy()
		self.ansatzCircuit = self.createAnsatzCircuit()
		UTheta = Operator(self.ansatzCircuit).data
		zeroQubit = np.zeros(2**self.n)
		zeroQubit[0] = 1
		u = np.matmul(UTheta,zeroQubit)  # |x(θ)⟩ = U(θ)|0⟩
		phi =  np.matmul(self.A,u)  # A|x(θ)⟩
		numerator = (np.matmul(self.b,phi))**2  # |⟨b|A|x(θ)⟩|²
		denominator = np.matmul(phi,phi)  # ⟨x(θ)|A†A|x(θ)⟩
		CG  = 1 - np.real(numerator/denominator)
		self.CostIterations.append(CG)
		return CG
	
	def costFunction(self,thetaParams):
		"""
		Quantum cost function evaluation using Hadamard tests
		======================================================
		C_G = 1 - |⟨b|A|x(θ)⟩|² / ⟨x(θ)|A†A|x(θ)⟩
		
		Numerator: |Σ_j α_j⟨b|P_j|x(θ)⟩|²
		- Requires O(n_Pauli) Hadamard test circuits
		- Each measurement: real + imaginary components
		
		Denominator: Σ_{j,k} α_j*α_k⟨x(θ)|P_j†P_k|x(θ)⟩
		- Requires O(n_Pauli²) Hadamard test circuits
		- Dominant cost of algorithm
		
		Total Measurements: O(n_Pauli² × shots) per cost evaluation
		Optimization typically requires 10-100 evaluations
		
		Cost Landscape:
		- Convex near solution (good for optimization)
		- May have barren plateaus (flat regions)
		- C_G = 0 when A|x(θ)⟩ = |b⟩ exactly
		"""
		# Compute numerator: g_j = ⟨b|P_j|x(θ)⟩
		gReal = np.zeros(self.nPauliTerms)
		gImag = np.zeros(self.nPauliTerms)
		self.ansatzParams = thetaParams.copy()
		self.ansatzCircuit = self.createAnsatzCircuit()
		self.ansatzMatrix = Operator(self.ansatzCircuit).data
		
		for j in range(self.nPauliTerms):
			# Real part
			circ1Real = self.HadamardTestCircuit1(self.ansatzMatrix,
										   self.PauliMatrices[j],self.UbMatrix)
			counts = simulateCircuit(circ1Real,shots = self.nShots)
			gReal[j] = (counts.get('0', 0)- counts.get('1', 0))/self.nShots
			
			# Imaginary part
			circ1Imag = self.HadamardTestCircuit1(self.ansatzMatrix,
										   self.PauliMatrices[j],self.UbMatrix,True)
			counts = simulateCircuit(circ1Imag,shots = self.nShots)
			gImag[j] = (counts.get('0', 0)- counts.get('1', 0))/self.nShots
		
		# Assemble numerator: |Σ_j α_j g_j|²
		numerator = 0
		for j in range(self.nPauliTerms):
			aj = self.PauliCoefficients[j]
			numerator = numerator + aj*(gReal[j]+1j*gImag[j])
		numerator = (numerator)**2
		
		# Compute denominator: h_{jk} = ⟨x(θ)|P_j†P_k|x(θ)⟩
		hReal = np.zeros((self.nPauliTerms,self.nPauliTerms))
		hImag =  np.zeros((self.nPauliTerms,self.nPauliTerms))
		
		for j in range(self.nPauliTerms):
			for k in range(self.nPauliTerms):
				# Real part
				circ2Real = self.HadamardTestCircuit2(self.ansatzMatrix, 
													 self.PauliMatrices[j],
													 self.PauliMatrices[k])
				counts = simulateCircuit(circ2Real,shots = self.nShots)
				hReal[j][k] = (counts.get('0', 0)- counts.get('1', 0))/self.nShots
				
				# Imaginary part
				circ2Imag = self.HadamardTestCircuit2(self.ansatzMatrix, 
													 self.PauliMatrices[j],
													 self.PauliMatrices[k],True)
				counts =  simulateCircuit(circ2Imag,shots = self.nShots)
				hImag[j][k] = (counts.get('0', 0)- counts.get('1', 0))/self.nShots
				
		# Assemble denominator: Σ_{j,k} α_j*α_k h_{jk}
		denominator = 0
		for j in range(self.nPauliTerms):
			aj = self.PauliCoefficients[j]
			for k in range(self.nPauliTerms):
				ak = self.PauliCoefficients[k]
				denominator = denominator + aj*ak*(hReal[j][k]+1j*hImag[j][k])

		# Compute cost
		CG  = 1 - np.real(numerator/denominator)
		self.CostIterations.append(CG)
		return CG
	
	def plotCostFunction(self,nSamples = 33):
		"""
		Plot cost function landscape (1-qubit systems only)
		===================================================
		Visualizes C_G(θ) for understanding optimization landscape.
		Helps identify local minima, barren plateaus, and convexity.
		"""
		if (self.n > 1):
			print('plotCostFunction is only valid for 1 qubit')
			return False
		if not self.dataOK:
			print('Check input data')
			return False
		thetaParams = np.zeros(self.nAnsatzParams)
		thetaValues = np.linspace(0,2*np.pi,nSamples)
		CGValues = []
		for theta in thetaValues:
			thetaParams[0] = theta
			CG = np.real(self.costFunction(thetaParams))
			CGValues.append(CG)
		plt.plot(thetaValues,CGValues)
		plt.xlabel(r'$\theta$')
		plt.ylabel(r'$C_G$')
		plt.grid(True)
		
	def executeVQLS(self):
		"""
		Main VQLS execution pipeline
		=============================
		
		Workflow:
		1. Initialize random parameters θ
		2. Optimize: minimize C_G(θ) using classical optimizer
		3. Extract solution from optimized ansatz
		4. Post-process: measure final state U(θ*)|0⟩
		
		Optimizer: COBYLA (Constrained Optimization BY Linear Approximation)
		- Derivative-free (good for noisy cost functions)
		- Handles bounds and constraints
		- Alternative: SPSA, Adam, or other gradient-based if shots high
		
		Returns: True if successful
		
		Result Available:
		- self.uVQLS: Solution state vector
		- self.minCG: Final cost value (should be ≈ 0)
		- self.thetaOptimal: Optimized parameters
		- self.CostIterations: Convergence history
		
		Complexity: O(iterations × n_Pauli² × shots)
		Typical: 20-100 iterations, depending on problem and ansatz
		"""
		if not self.dataOK:
			print('Check input data')
			return False

		self.CostIterations = []
		thetaParams = np.random.rand(self.nAnsatzParams)
		
		# Classical optimization of cost function
		result = scipy.optimize.minimize(self.costFunction,
								   thetaParams, method='COBYLA' )
		
		self.minCG = result['fun']
		self.thetaOptimal = result['x']
		
		# Extract solution state by measuring optimized ansatz
		self.ansatzParams = self.thetaOptimal.copy()
		self.ansatzCircuit = self.createAnsatzCircuit(True)
		counts = simulateCircuit(self.ansatzCircuit, shots=self.nShots)
		
		# Reconstruct state from measurement statistics
		self.uVQLS = 0
		for key in counts:
			v = np.real(Statevector.from_label(key))
			self.uVQLS = self.uVQLS + v*np.sqrt(counts[key]/self.nShots)
		
		return True

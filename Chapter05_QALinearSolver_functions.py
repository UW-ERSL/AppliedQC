"""
Quantum Annealing for QUBO Problems
====================================
Implements a box-method solver for Quadratic Unconstrained Binary Optimization (QUBO)
problems using quantum annealing, simulated annealing, or exact solvers.

Key Algorithm: Adaptive Box Method
- Iteratively refines solution by translating and contracting search space
- Encodes continuous variables using binary qubits via discretization
- Solves Ax=b by minimizing H = 0.5*x^T*A*x - x^T*b

References:
- Suresh, S. and Suresh, K., 2025. Optimal box contraction for solving linear systems via simulated and quantum annealing. Engineering Optimization, 57(9), pp.2597-2608.
- Kadowaki & Nishimori (1998): Quantum annealing in transverse Ising model
- Lucas (2014): Ising formulations of many NP problems

- PyQUBO documentation for symbolic QUBO construction
"""
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

from pyqubo import Binary, Array,Placeholder
from dimod.reference.samplers import ExactSolver
import neal
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite


import numpy as np
import matplotlib.pyplot as plt
from pyqubo import Binary, Array
from dimod.reference.samplers import ExactSolver
import neal

class TrussQUBOOptimizer:
    """
    QUBO-based truss topology optimization using quantum annealing.
    Combines QUBO formulation with FEM constraint evaluation.
    """
    
    def __init__(self, fem_model, loads, fixed_dofs, 
                 penalty_min_members=100.0,
                 penalty_infeasible=1000.0):
        """
        Parameters:
        -----------
        fem_model : TrussFEM
            Finite element model
        loads : ndarray
            Load vector
        fixed_dofs : list
            Fixed DOF indices
        penalty_min_members : float
            Penalty weight for minimum member constraint
        penalty_infeasible : float
            Penalty for FEM constraint violations
        """
        self.fem = fem_model
        self.loads = loads
        self.fixed_dofs = fixed_dofs
        self.N = len(fem_model.elements)
        self.penalty_min = penalty_min_members
        self.penalty_infeas = penalty_infeasible
        
        # Precompute member lengths and weights
        self.lengths = np.zeros(self.N)
        for idx, (i, j) in enumerate(self.fem.elements):
            dx = self.fem.nodes[j, 0] - self.fem.nodes[i, 0]
            dy = self.fem.nodes[j, 1] - self.fem.nodes[i, 1]
            self.lengths[idx] = np.sqrt(dx**2 + dy**2)
        
        self.weights = self.fem.rho * self.fem.A * self.lengths
        
    def build_qubo(self, target_members=None, adaptive_penalty=None):
        """
        Build QUBO Hamiltonian for truss optimization.
        
        H = Σ w_i * x_i + λ1 * (Σ x_i - M_min)^2 + λ2 * constraint_violations
        
        Parameters:
        -----------
        target_members : int, optional
            Target number of members (for penalty tuning)
        adaptive_penalty : dict, optional
            Adaptive penalties based on previous iterations
            
        Returns:
        --------
        model : pyqubo model
        x : Array of Binary variables
        """
        # Create binary variables for each potential member
        x = Array.create('x', shape=self.N, vartype='BINARY')
        
        # Objective: minimize total weight
        H_weight = sum(self.weights[i] * x[i] for i in range(self.N))
        
        # Constraint: minimum number of members
        # For 2D truss: M_min = 2n - 3 where n is number of nodes
        min_members = max(10, 2 * self.fem.n_nodes - 3)
        if target_members is not None:
            min_members = target_members
        
        # Penalty for deviation from minimum members
        # (Σ x_i - M_min)^2 = Σ x_i^2 - 2*M_min*Σ x_i + M_min^2
        # Since x_i^2 = x_i for binary: Σ x_i - 2*M_min*Σ x_i + constant
        member_sum = sum(x[i] for i in range(self.N))
        H_members = self.penalty_min * (member_sum - min_members)**2
        
        # Adaptive penalties from previous infeasible solutions
        H_adaptive = 0
        if adaptive_penalty is not None:
            for idx, penalty in adaptive_penalty.items():
                # Penalize members that caused violations
                H_adaptive += penalty * (1 - x[idx])  # Encourage removal
        
        # Total Hamiltonian
        H = H_weight + H_members + H_adaptive
        
        # Compile to BQM
        model = H.compile()
        
        return model, x
    
    def evaluate_qubo_solution(self, sample, u_hat=0.01, sigma_hat=250e6):
        """
        Evaluate a QUBO solution using FEM.
        
        Parameters:
        -----------
        sample : dict
            Binary solution from QUBO solver
        u_hat, sigma_hat : float
            Constraint limits
            
        Returns:
        --------
        metrics : dict
            FEM evaluation results
        design : ndarray
            Design vector
        """
        # Extract design from sample
        design = np.array([sample[f'x[{i}]'] for i in range(self.N)])
        
        # Evaluate with FEM
        metrics = self.fem.evaluate_design(
            design, self.loads, self.fixed_dofs,
            u_hat=u_hat, sigma_hat=sigma_hat
        )
        
        return metrics, design
    
    def solve_exact(self, u_hat=0.01, sigma_hat=250e6):
        """
        Solve using exact QUBO solver (only for small N < 20).
        
        Returns:
        --------
        best_design : ndarray
            Optimal design found
        best_metrics : dict
            Performance metrics
        all_results : list
            All feasible solutions found
        """
        print(f"Building QUBO for {self.N} members...")
        model, x = self.build_qubo()
        bqm = model.to_bqm()
        
        print("Solving with ExactSolver...")
        sampler = ExactSolver()
        sampleset = sampler.sample(bqm)
        
        print(f"\nEvaluating {len(sampleset)} QUBO solutions with FEM...")
        
        best_weight = np.inf
        best_design = None
        best_metrics = None
        all_results = []
        
        for i, (sample, energy) in enumerate(zip(sampleset.samples(), 
                                                   sampleset.data(['energy']))):
            metrics, design = self.evaluate_qubo_solution(sample, u_hat, sigma_hat)
            
            if metrics['feasible']:
                all_results.append({
                    'design': design.copy(),
                    'metrics': metrics,
                    'qubo_energy': energy
                })
                
                if metrics['weight'] < best_weight:
                    best_weight = metrics['weight']
                    best_design = design.copy()
                    best_metrics = metrics
            
            if (i + 1) % 100 == 0:
                print(f"  Evaluated {i+1}/{len(sampleset)} solutions, "
                      f"found {len(all_results)} feasible")
        
        print(f"\nFound {len(all_results)} feasible designs")
        if best_design is not None:
            print(f"Best design: {np.sum(best_design)} members, "
                  f"weight = {best_weight:.2f} kg")
        
        return best_design, best_metrics, all_results
    
    def solve_annealing(self, num_reads=100, num_iterations=1, 
                       u_hat=0.01, sigma_hat=250e6):
        """
        Solve using simulated annealing with iterative refinement.
        
        Parameters:
        -----------
        num_reads : int
            Number of annealing runs per iteration
        num_iterations : int
            Number of refinement iterations
        u_hat, sigma_hat : float
            Constraint limits
            
        Returns:
        --------
        best_design : ndarray
            Best design found
        best_metrics : dict
            Performance metrics
        history : list
            Optimization history
        """
        print(f"Solving with Simulated Annealing...")
        print(f"  {num_iterations} iterations × {num_reads} reads/iteration\n")
        
        best_overall_design = None
        best_overall_weight = np.inf
        best_overall_metrics = None
        history = []
        
        adaptive_penalty = None
        
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            
            # Build QUBO (with adaptive penalties if available)
            model, x = self.build_qubo(adaptive_penalty=adaptive_penalty)
            bqm = model.to_bqm()
            
            # Sample with simulated annealing
            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=num_reads)
            
            # Evaluate unique solutions
            unique_samples = []
            seen_designs = set()
            
            for sample in sampleset.samples():
                design_tuple = tuple(sample[f'x[{i}]'] for i in range(self.N))
                if design_tuple not in seen_designs:
                    unique_samples.append(sample)
                    seen_designs.add(design_tuple)
            
            print(f"  Evaluating {len(unique_samples)} unique solutions...")
            
            iteration_best_weight = np.inf
            iteration_best_design = None
            feasible_count = 0
            
            for sample in unique_samples:
                metrics, design = self.evaluate_qubo_solution(sample, 
                                                             u_hat, sigma_hat)
                
                if metrics['feasible']:
                    feasible_count += 1
                    
                    if metrics['weight'] < iteration_best_weight:
                        iteration_best_weight = metrics['weight']
                        iteration_best_design = design.copy()
                    
                    if metrics['weight'] < best_overall_weight:
                        best_overall_weight = metrics['weight']
                        best_overall_design = design.copy()
                        best_overall_metrics = metrics
                        print(f"  ✓ New best: {np.sum(design)} members, "
                              f"weight = {metrics['weight']:.2f} kg")
            
            print(f"  Found {feasible_count} feasible designs")
            
            history.append({
                'iteration': iteration,
                'feasible_count': feasible_count,
                'best_weight': iteration_best_weight,
                'best_design': iteration_best_design
            })
            
            # Update adaptive penalties for next iteration
            # (Simple strategy: penalize infeasible solutions)
            if iteration < num_iterations - 1 and feasible_count < num_reads // 10:
                print(f"  Low feasibility - adjusting penalties")
                # This is a placeholder - more sophisticated strategies possible
        
        print(f"\n{'='*60}")
        if best_overall_design is not None:
            print(f"Best design found:")
            print(f"  Members: {np.sum(best_overall_design)}")
            print(f"  Weight: {best_overall_weight:.2f} kg")
            print(f"  Max displacement: {best_overall_metrics['max_disp']*1000:.4f} mm")
            print(f"  Max stress: {best_overall_metrics['max_stress']/1e6:.2f} MPa")
            print(f"  Active members: {np.where(best_overall_design)[0].tolist()}")
        else:
            print("No feasible design found!")
        print(f"{'='*60}\n")
        
        return best_overall_design, best_overall_metrics, history



class QUBOBoxSolverClass:
	"""
	Adaptive Box Method for QUBO-based Linear System Solving
	=========================================================
	Solves Ax = b using quantum/classical annealing within adaptive search boxes.
	
	Algorithm: Box method with binary encoding
	- Each continuous variable x_i encoded as: x_i = c_i + L*(-2*q_0 + q_1)
	- Two qubits per dimension provide ternary discretization {-2L, 0, L}
	- Box contracts by factor beta when no improvement found
	- Complexity: O(iterations × sampling_cost)
	
	Parameters:
	-----------
	beta : float (default 0.5)
		Box contraction factor (0 < beta < 1). Smaller values = faster convergence but risk missing solutions.
	LBox0 : float (default 1)
		Initial box half-width. Should span expected solution magnitude.
	tol : float (default 1e-6)
		Relative tolerance for convergence: L/LBox0 < tol
	samplingMethod : str
		"exact" : Exhaustive search (exponential cost, use for n < 20)
		"simulatedAnnealing" : Classical SA (Neal sampler)
		"hybridQuantumAnnealing" : D-Wave hybrid classical-quantum
		"quantumAnnealing" : Full quantum annealing on D-Wave hardware
	nSamples : int (default 50)
		Number of annealing runs per iteration (classical/quantum samplers only)
	boxMaxIteration : int (default 100)
		Maximum box iterations before termination
	"""
	def __init__(self,beta=0.5,LBox0 = 1,tol = 1e-6,
			  samplingMethod = "simulatedAnnealing", 
			  nSamples = 50,boxMaxIteration = 100):	
		
		# Store optimization parameters
		self.beta = beta
		self.LBox0 = LBox0
		self.boxMaxIteration = boxMaxIteration
		self.relativeTolerance = tol
		self.samplingMethod = samplingMethod
		self.nSamples = nSamples
		
		# Initialize appropriate sampler based on method
		# Exact solver: O(2^n) complexity, only feasible for small problems
		if (self.samplingMethod == "exact"):
			self.sampler = ExactSolver()
		# Simulated annealing: Classical approximation, O(nSamples × problem_size)
		elif (self.samplingMethod == "simulatedAnnealing"):
			self.sampler = neal.SimulatedAnnealingSampler()
		# Hybrid: Combines classical and quantum resources for larger problems
		elif (self.samplingMethod == "hybridQuantumAnnealing"):
			self.sampler = LeapHybridSampler()
		# Full quantum annealing: Requires D-Wave hardware access
		elif (self.samplingMethod == "quantumAnnealing"):
			self.sampler = EmbeddingComposite(DWaveSampler())
		else:
			print("Invalid sampling method")

		# Binary encoding: 2 qubits per dimension gives 4 states, using 3: {-2L, 0, L}
		self.nQubitsPerDimension = 2  # Fixed for ternary encoding
		
	def modelWithPlaceHolders(self):
		"""
		Construct symbolic QUBO model using PyQUBO placeholders
		========================================================
		Creates reusable model that can be fed different A, b values without recompilation.
		
		Encoding scheme:
		- Each variable x_i = c_i + L*(-2*q_i[0] + q_i[1])
		- Placeholders A[i][j], b[i], c[i], L are substituted at solve time
		
		Hamiltonian: H = Σ_i x_i * (0.5 * Σ_j A[i][j]*x_j) - Σ_i x_i*b[i]
		This represents the quadratic form: 0.5*x^T*A*x - x^T*b
		
		Efficiency: Compile once, solve many times with different parameters
		"""
		# Create binary variable arrays: q[i][j] where i=dimension, j∈{0,1} are the two qubits
		q = self.matrixSize*[None]
		for i in range(self.matrixSize):
			q[i] = Array.create("q[" + str(i)+"]",shape = self.nQubitsPerDimension,vartype = "BINARY")
		
		# Initialize placeholders for problem parameters (to be fed at solve time)
		c = self.matrixSize*[0]  # Box center coordinates
		b = self.matrixSize*[0]  # RHS vector
		A = self.matrixSize*[0]  # System matrix
		x = self.matrixSize*[0]  # Symbolic continuous variables (functions of qubits)
		for i in range(self.matrixSize):
			A[i] = self.matrixSize*[0]
		L = Placeholder('L')  # Box half-width (adaptive parameter)
		
		# Create placeholder symbols
		for i in range(self.matrixSize):
			c[i] = Placeholder('c[%d]' %i)
			b[i] = Placeholder('b[%d]' %i)
			for j in range(self.matrixSize):
				A[i][j] = Placeholder("A[{i}][{j}]".format(i = i, j = j))
		
		# Encode continuous variables via binary qubits
		# x[i] ∈ {c[i]-2L, c[i], c[i]+L} based on qubit states
		for i in range(self.matrixSize):
			x[i] = c[i] + L*(-2*q[i][0] + q[i][1])
		   
		# Construct quadratic Hamiltonian: H = 0.5*x^T*A*x - x^T*b
		# Minimizing H solves the linear system approximately within the current box
		H = 0
		for  i in range(self.matrixSize):
			Ax = 0
			for j in range(self.matrixSize):
				Ax = Ax + A[i][j]*x[j]
			H = H + x[i]*(0.5*Ax) - x[i]*b[i]
		
		# Compile to QUBO form (symbolic, fast to evaluate with different parameters)
		self.model = H.compile()
		return self.model
	
	def plotBox(self,center,L,iteration):
		"""
		Visualize box evolution for 2D problems
		========================================
		Plots the current search box for debugging/visualization.
		Only functional for 2D systems (matrixSize=2).
		
		Box corners: [c-2L, c+L] × [c-2L, c+L]
		Color cycles through palette to distinguish iterations.
		"""
		if (self.matrixSize != 2):  # Only visualize 2D case
			return
	   
		plotColors = ['k','r','b','g','c','m','y']
		index = iteration % len(plotColors)
		# Define box vertices based on encoding: {c-2L, c, c+L}
		xBox = [center[0]-2*L, center[0]+L,center[0]+L,center[0]-2*L,center[0]-2*L ]
		yBox = [center[1]-2*L, center[1]-2*L, center[1]+L, center[1]+L,center[1]-2*L]
		plt.plot(xBox,yBox,plotColors[index])  
	 
		plt.savefig("./results/" + str(iteration) +".png")
		
	def QUBOBoxSolve(self,A, b,xGuess = [],debug = False):
		"""
		Main solver: Adaptive box method for Ax = b using quantum/classical annealing
		==============================================================================
		
		Algorithm:
		1. Initialize box centered at xGuess (or origin) with half-width LBox0
		2. For each iteration:
		   a. Construct QUBO problem within current box
		   b. Sample using chosen annealing method
		   c. If improved solution found: translate box center to new optimum
		   d. If no improvement: contract box by factor beta
		3. Terminate when L/LBox0 < tolerance or max iterations reached
		
		Parameters:
		-----------
		A : ndarray (n × n)
			System matrix (should be symmetric positive definite for convex problem)
		b : ndarray (n,)
			Right-hand side vector
		xGuess : list/ndarray (optional)
			Initial guess for solution (default: zero vector)
		debug : bool
			Enable visualization and verbose output
			
		Returns:
		--------
		[x_solution, final_L, iterations, success, n_translations, n_contractions, results]
		- x_solution: Best approximation to solution
		- final_L: Final box size (indicator of convergence quality)
		- iterations: Number of box iterations performed
		- success: Boolean indicating convergence within tolerance
		- n_translations: Count of successful box moves
		- n_contractions: Count of box contractions
		- results: Final annealing results object
		
		Complexity: O(iterations × sampling_complexity)
		- Exact: O(iterations × 2^(2n)) where n = problem size
		- Annealing: O(iterations × nSamples × problem_size)
		"""
		self.matrixSize = A.shape[0]
		self.model = self.modelWithPlaceHolders()
		
		# Initialize qubit solution array
		qSol = self.matrixSize*[None]
		for i in range(self.matrixSize):
			qSol[i] = self.nQubitsPerDimension*[0]
		
		# Set initial box center
		if (len(xGuess) == 0 ):
			center = self.matrixSize*[0]  # Default to origin
		else:
			center = xGuess
		
		# Store problem parameters in dictionary for PyQUBO
		self.modelDictionary = {}
		for  i in range(self.matrixSize):
			self.modelDictionary['b[%d]' %i] = b[i]
			for j in range(self.matrixSize):
				self.modelDictionary["A[{i}][{j}]".format(i = i, j = j)] = A[i,j]
		
		# Initialize box size and convergence tracking
		L = self.LBox0
		boxSuccess = True
		nTranslations = 0   # Successful moves
		nContractions = 0   # Box size reductions
		PEHat = 0  # Best potential energy found so far
		
		# Main box iteration loop
		for iteration in range(self.boxMaxIteration):
			# Convergence check: box small enough relative to initial size
			if (L/self.LBox0 < self.relativeTolerance):
				break
			if (iteration == self.boxMaxIteration):
				break
			
			# Update box parameters in model
			self.modelDictionary['L'] =  L
			for  i in range(self.matrixSize):
				self.modelDictionary['c[%d]' %i] = center[i]
			
			# Convert symbolic model to Binary Quadratic Model with current parameters
			bqm = self.model.to_bqm(feed_dict = self.modelDictionary)
			
			# Sample using selected method
			# Exact solver: guarantees global optimum but exponential cost
			if (self.samplingMethod == "exact"):
				results = self.sampler.sample(bqm)
			# Annealing methods: heuristic, polynomial cost
			elif (self.samplingMethod == "simulatedAnnealing"):
				results = self.sampler.sample(bqm, num_reads=self.nSamples)
			elif (self.samplingMethod == "openjijAnnealing"):
				results = self.sampler.sample(bqm, num_reads=self.nSamples)
			elif (self.samplingMethod == "quantumAnnealing"):
				results = self.sampler.sample(bqm)

			# Extract best sample and its energy
			sample = results.first.sample
			PEStar = results.first.energy 
			
			# Decision: translate or contract?
			# Translate if we found improvement (with small tolerance for numerical noise)
			if (PEStar < PEHat*(1+1e-8)):  # Improvement found
				# Extract qubit values from sample
				for i in range(self.matrixSize):		
					qSol[i][0]= sample["q["+str(i)+"][0]"]
					qSol[i][1]= sample["q["+str(i)+"][1]"]
				PEHat = PEStar
				# Translate box center to new solution
				# Decode: x_i = c_i + L*(-2*q[0] + q[1])
				for i in range(self.matrixSize):  
					center[i] = center[i] + L*(-2*qSol[i][0] + qSol[i][1])
				nTranslations = nTranslations + 1
			else:  # No improvement: contract box
				L = L*self.beta
				nContractions = nContractions + 1
			
			# Optional debugging output
			if(debug):
				self.plotBox(center,L,iteration)
				print('Iter: ' + str(iteration)  + '; center: ' + str(center) + '; PE: ' + str(PEStar) + '; L: ' + str(L))
		
		# Check final convergence
		if ( L/self.LBox0  > self.relativeTolerance):
			print("Box method did not converge to desired tolerance")
			boxSuccess = False
	
		return [np.array(center),L,iteration,boxSuccess,nTranslations,nContractions,results]
	

class SimulatedQuantumAnnealing:
    def __init__(self, Q, M=10, Gamma_init=1.0):
        """
        Q: QUBO matrix
        M: Number of Trotter slices (imaginary time discretization)
        Gamma_init: Initial transverse field strength
        """
        self.Q = Q
        self.N = Q.shape[0]
        self.M = M  # Trotter slices
        self.Gamma = Gamma_init
        
    def transverse_field_energy(self, spins):
        """Energy from transverse field (tunneling term)."""
        # Coupling between adjacent Trotter slices
        energy = 0
        for m in range(self.M):
            for i in range(self.N):
                # Flip energy between slice m and m+1
                if spins[m, i] != spins[(m+1) % self.M, i]:
                    energy -= self.Gamma
        return energy
    
    def problem_energy(self, spins):
        """Classical QUBO energy."""
        return sum(spins[m] @ self.Q @ spins[m] for m in range(self.M)) / self.M
    
    def anneal(self, steps=1000, beta_init=0.1, beta_final=10):
        """Run SQA with annealing schedule."""
        # Initialize M replicas
        spins = np.random.choice([-1, 1], size=(self.M, self.N))
        Gamma = self.Gamma
        for step in range(steps):
            # Anneal both temperature and transverse field
            beta = beta_init + (beta_final - beta_init) * step / steps
            Gamma = Gamma * (1 - step / steps)  # Decrease tunneling
            
            # Monte Carlo sweep
            for m in range(self.M):
                for i in range(self.N):
                    # Propose spin flip
                    delta_E = self._flip_energy(spins, m, i, Gamma)
                    
                    if delta_E < 0 or np.random.rand() < np.exp(-beta * delta_E):
                        spins[m, i] *= -1
        
        # Return ground state from first Trotter slice
        return (spins[0] + 1) // 2  # Convert to {0,1}
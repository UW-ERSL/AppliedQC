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
- Kadowaki & Nishimori (1998): Quantum annealing in transverse Ising model
- Lucas (2014): Ising formulations of many NP problems

- PyQUBO documentation for symbolic QUBO construction
"""
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer


import numpy as np
import matplotlib.pyplot as plt
from pyqubo import Binary, Array
from dimod.reference.samplers import ExactSolver
import neal




import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

"""
Quantum Annealing Topology Optimization - Simplified Notation
Based on direct area updates without intermediate design variables

Notation matches the LaTeX formulation:
- A_e^(i): Area of element e at iteration i
- α_e^(i): Updater variable (growth factor)
- E_e^(i-1): Element strain energy before update
- q_e: Binary qubit variable for element e
- q_s: Binary qubit variable for slack
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import time


class QATrussOptimizer:
    """
    Quantum Annealing Topology Optimizer using simplified notation
    References:Sukulthanasorn, et. al 2025. A novel design update framework for topology 
    optimization with quantum annealing: Application to truss and continuum structures. 
    Computer Methods in Applied Mechanics and Engineering, 437, p.117746.
    """
    
    def __init__(self, 
                 fem_model,
                 V_bar: float,              # Target volume
                 alpha_under: float = 0.9,  # Lower bound for updater
                 alpha_over: float = 1.1,   # Upper bound for updater
                 S_under: float = 0.0,      # Lower bound for slack
                 S_over: float = 0.05,      # Upper bound for slack
                 lambda_penalty: float = 100.0,
                 A_min: float = 1e-6,       # Minimum area (numerical stability)
                 A_max: float = 10.0,       # Maximum area (optional bound)
                 verbose: bool = False):
        """
        Initialize quantum optimizer
        
        Parameters:
        -----------
        fem_model : TrussFEM
            Truss finite element model
        V_bar : float
            Target total volume constraint
        alpha_under : float
            Lower bound for updater (q=0 gives this)
        alpha_over : float
            Upper bound for updater (q=1 gives this)
        S_under : float
            Lower bound for slack variable
        S_over : float
            Upper bound for slack variable
        lambda_penalty : float
            Penalty parameter λ for volume constraint
        A_min : float
            Minimum area to prevent singularity
        A_max : float
            Maximum area (optional, set large if unbounded)
        verbose : bool
            Print detailed progress
        """
        self.fem = fem_model
        self.V_bar = V_bar
        self.alpha_under = alpha_under
        self.alpha_over = alpha_over
        self.S_under = S_under
        self.S_over = S_over
        self.lambda_penalty = lambda_penalty
        self.A_min = A_min
        self.A_max = A_max
        self.verbose = verbose
        
        # Problem dimensions
        self.n_elements = fem_model.n_elements
        
        # Derived quantities
        self.Delta_alpha = alpha_over - alpha_under
        self.Delta_S = S_over - S_under
        
        # Element geometry (constant)
        self.L = fem_model.L.copy()    # Element lengths
        self.E = fem_model.E           # Young's modulus
        
        # Current areas (state variable)
        self.A = fem_model.A.copy()    # A_e^(i)
        
        # Initial reference values
        self.E0 = None  # Will be set in first iteration
        self.V0 = np.sum(self.A * self.L)  # Initial volume
        
        # History tracking
        self.objective_history = []
        self.volume_history = []
        self.area_history = []
        self.iteration = 0
        
    def update_fem_areas(self):
        """Update FEM model with current areas"""
        self.fem.set_area(self.A)
    
    def get_element_energies(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Solve FEM and compute element strain energies
        
        Computes: E_e^(i-1) = (d_e^(i))^T K_e^(i-1) d_e^(i)
        
        Returns:
        --------
        element_energies : ndarray
            E_e^(i-1) for each element
        displacements : ndarray
            Global displacement vector d^(i)
        valid : bool
            Whether FEM solution is valid
        """
        # Solve FEM with current areas
        d, valid = self.fem.solve()
        
        if not valid:
            if self.verbose:
                print(" Warning: FEM solution not valid!")
            return None, None, False
        
        # Store displacements in FEM
        self.fem.displacements = d
        
        # Compute element strain energies
        element_energies = np.zeros(self.n_elements)
        
        for elem_idx in range(self.n_elements):
            # Get element displacement vector d_e^(i)
            d_elem = self.fem.get_element_displacements(elem_idx)
            
            # Get element stiffness matrix K_e^(i-1)
            K_elem = self.fem.get_element_stiffness_matrix(elem_idx)
            
            # Compute element energy: E_e^(i-1) = d_e^T K_e d_e
            element_energies[elem_idx] = d_elem.T @ K_elem @ d_elem
        
        return element_energies, d, valid
    
    def formulate_qubo(self, element_energies: np.ndarray) -> Dict:
        """
        Formulate QUBO problem with simplified notation
        
        QUBO coefficients:
        - Q_ee: Diagonal terms for element variables
        - Q_ej: Off-diagonal coupling between elements
        - Q_ss: Diagonal term for slack variable
        - Q_es: Coupling between elements and slack
        
        Parameters:
        -----------
        element_energies : ndarray
            E_e^(i-1) for each element
            
        Returns:
        --------
        Q : dict
            QUBO coefficients {(i,j): Q_ij}
        """
        # Initialize QUBO dictionary
        Q = {}
        
        # Set initial energy reference (first iteration only)
        if self.E0 is None:
            self.E0 = np.sum(element_energies)
            if self.verbose:
                print(f"  Initial strain energy E^0 = {self.E0:.4e}")
        
        # Normalized quantities
        E_star = element_energies / self.E0  # E_e* = E_e^(i-1) / E^0
        V_current = self.A * self.L          # V_e^(i-1)
        v_star = V_current / self.V_bar      # v_e* = V_e^(i-1) / V_bar
        
        # Constant term C
        C = self.alpha_under * np.sum(v_star) + self.S_under - 1.0
        
        for e in range(self.n_elements):
            var_idx = e
            
            # Objective contribution: -Δα·E_e*
            obj_term = -self.Delta_alpha * E_star[e]
            
            # Constraint contribution: λ[(Δα)²(v_e*)² + 2Δα·C·v_e*]
            constraint_term = self.lambda_penalty * (
                (self.Delta_alpha**2) * (v_star[e]**2) +
                2 * self.Delta_alpha * C * v_star[e]
            )
            
            Q[(var_idx, var_idx)] = obj_term + constraint_term
        

        for e in range(self.n_elements):
            for j in range(e + 1, self.n_elements):
                var_e = e
                var_j = j
                
                coeff = 2 * self.lambda_penalty * (self.Delta_alpha**2) * v_star[e] * v_star[j]
                Q[(var_e, var_j)] = coeff

        var_s = self.n_elements  # Slack variable index
        
        Q[(var_s, var_s)] = self.lambda_penalty * (
            self.Delta_S**2 + 
            2 * self.Delta_S * C
        )

        for e in range(self.n_elements):
            var_e = e
            coeff = 2 * self.lambda_penalty * self.Delta_alpha * self.Delta_S * v_star[e]
            Q[(var_e, var_s)] = coeff
        
        return Q
    
    def decode_solution(self, q_solution: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Decode binary QUBO solution to updaters
        
        α_e = (1-q_e)·α_under + q_e·α_over = α_under + q_e·Δα
        S = (1-q_s)·S_under + q_s·S_over = S_under + q_s·ΔS
        
        Parameters:
        -----------
        q_solution : ndarray
            Binary solution vector [q_1, ..., q_N, q_s]
            
        Returns:
        --------
        alpha : ndarray
            Updater values for each element
        slack : float
            Slack variable value
        """
        # Extract element qubits
        q_elements = q_solution[:self.n_elements]
        
        # Extract slack qubit
        q_slack = q_solution[self.n_elements]
        
        # Decode updaters: α_e = α_under + q_e·Δα
        alpha = self.alpha_under + q_elements * self.Delta_alpha
        
        # Decode slack: S = S_under + q_s·ΔS
        slack = self.S_under + q_slack * self.Delta_S
        
        return alpha, slack
    
    def update_design(self, alpha: np.ndarray):
        """
        Update areas using decoded updaters
        
        A_e^(i) = α_e^(i) · A_e^(i-1)
        
        Parameters:
        -----------
        alpha : ndarray
            Updater values from decoded QUBO solution
        """
        # Multiplicative update
        self.A = alpha * self.A
        
        # Enforce bounds [A_min, A_max]
        self.A = np.clip(self.A, self.A_min, self.A_max)
        
        # Update FEM model
        self.update_fem_areas()
        
        # Store history
        self.area_history.append(self.A.copy())

    
    def qubo_solver(self, Q: Dict) -> np.ndarray:
        """
        Solve QUBO problem using simulated annealing
        
        Parameters:
        -----------
        Q : dict
            QUBO coefficients {(i,j): Q_ij}
            
        Returns:
        --------
        q_solution : ndarray
            Binary solution vector
        """
        # Use neal's Simulated Annealing Sampler
        sampler = neal.SimulatedAnnealingSampler()
        
        # Sample from QUBO
        sampleset = sampler.sample_qubo(Q, 
                                        num_reads=500,      # More samples
                                        num_sweeps=2000,    # More sweeps
                                        verbose=False)
        
        # Get best solution
        best_sample = sampleset.first.sample
        
        # Convert to ndarray
        n_vars = self.n_elements + 1  # Elements + slack
        q_solution = np.array([best_sample[i] for i in range(n_vars)])
    
        
        return q_solution
    
    def optimize(self, 
                 max_iterations: int = 50,
                 tolerance: float = 0.01,
                 convergence_window: int = 5) -> Dict:
        """
        Main optimization loop
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of design iterations
        tolerance : float
            Convergence tolerance for objective function change
        convergence_window : int
            Number of iterations to check for convergence
            
        Returns:
        --------
        results : dict
            Optimization results
        """
        print("="*70)
        print("QUANTUM ANNEALING TOPOLOGY OPTIMIZATION")
        print("="*70)
        print(f"Problem: {self.n_elements} elements")
        print(f"Target volume: {self.V_bar:.4f} m³")
        print(f"Initial areas: {self.A}")
        print("="*70)
        
        start_time = time.time()
        
        for iteration in range(max_iterations):
            self.iteration = iteration
            iter_start = time.time()
            
            element_energies, displacements, valid = self.get_element_energies()
            
            if not valid:
                print(" FEM solution failed! Stopping.")
                break
            
            # Evaluate current design
            metrics = self.fem.evaluate_design(desiredVolumeFraction=self.V_bar/self.V0)
            compliance = metrics['compliance']
            volume = metrics['volume']
            
            self.objective_history.append(compliance)
            self.volume_history.append(volume)
            
            Q = self.formulate_qubo(element_energies)
            
            q_solution = self.qubo_solver(Q)
         
            alpha, slack = self.decode_solution(q_solution)

            self.update_design(alpha)
            
            iter_time = time.time() - iter_start
            
            if iteration >= convergence_window:
                recent_obj = self.objective_history[-convergence_window:]
                obj_change = abs(recent_obj[-1] - recent_obj[0]) / abs(recent_obj[0])
                
                if obj_change < tolerance:
                    print(f"\n{'='*70}")
                    print(f"✓ CONVERGED after {iteration + 1} iterations")
                    print(f"  Objective change < {tolerance:.2%} for {convergence_window} iterations")
                    print(f"{'='*70}")
                    break
            
            if self.verbose:
                print(f"  Iteration time: {iter_time:.3f}s")
        
        total_time = time.time() - start_time
        
        # Final evaluation
        final_metrics = self.fem.evaluate_design(desiredVolumeFraction=self.V_bar/self.V0)


        print(f"\n{'='*70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total iterations: {iteration + 1}")
        print(f"Total time: {total_time:.2f}s ({total_time/(iteration+1):.3f}s per iteration)")
        print(f"\nFinal Design:")
        print(f"  Compliance: {final_metrics['compliance']:.4e} J")
        print(f"  Volume: {final_metrics['volume']:.4f} m³ (target: {self.V_bar:.4f})")
        print(f"  Active members: {np.sum(self.A > 100*self.A_min)}/{self.n_elements}")
        print(f"  Area range: [{self.A.min():.4e}, {self.A.max():.4e}] m²")
        print(f"  Feasible: {final_metrics['feasible']}")
        print(f"{'='*70}")
        
        # Package results
        results = {
            'A_final': self.A,
            'area_history': self.area_history,
            'objective_history': self.objective_history,
            'volume_history': self.volume_history,
            'final_metrics': final_metrics,
            'iterations': iteration + 1,
            'total_time': total_time,
            'converged': iteration < max_iterations - 1
        }
        
        return results
    
    def plot_convergence(self):
        """Plot optimization convergence history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Compliance history
        ax = axes[0]
        ax.plot(self.objective_history, 'b-o', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Compliance (J)', fontsize=12)
        ax.set_title('Objective Function Convergence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Volume history
        ax = axes[1]
        ax.plot(self.volume_history, 'r-o', linewidth=2, markersize=6)
        ax.axhline(y=self.V_bar, color='k', linestyle='--', label=f'Target = {self.V_bar:.2f} m³')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Volume (m³)', fontsize=12)
        ax.set_title('Volume Constraint', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.show()

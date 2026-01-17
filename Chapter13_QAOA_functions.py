import numpy as np
import time
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class QAOATrussOptimizer:
    def __init__(self, fem_model, V_bar, alpha_under=0.9, alpha_over=1.1, 
                 S_under=0.0, S_over=0.05, lambda_penalty=100.0, 
                 use_exact = True,
                 p_layers=2, max_iterations=50, verbose=True):
        """
        QAOA implementation following the logic of the Quantum Annealing solver.
        Uses a multiplicative update rule: A_new = alpha * A_old.
        """
        self.fem = fem_model
        self.V_bar = V_bar
        self.alpha_under = alpha_under
        self.alpha_over = alpha_over
        self.S_under = S_under
        self.S_over = S_over
        self.lambda_penalty = lambda_penalty
        self.p_layers = p_layers
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.use_exact = use_exact # New toggle
        
        self.n_elements = fem_model.n_elements
        self.n_qubits = self.n_elements + 1  # 1 bit per element + 1 slack bit
        
        self.Delta_alpha = alpha_over - alpha_under
        self.Delta_S = S_over - S_under
        self.E0 = None  # Reference energy for normalization
        
        # Initial QAOA parameters (gammas and betas)
        self.optimal_params = np.random.rand(2 * self.p_layers)
        self.earlyStop = False

    def get_element_energies(self):
        """Solve FEM and compute element strain energies."""
        d, valid = self.fem.solve()
        if not valid: return None, None, False
        
        self.fem.displacements = d
        element_energies = np.zeros(self.n_elements)
        for e in range(self.n_elements):
            d_e = self.fem.get_element_displacements(e)
            K_e = self.fem.get_element_stiffness_matrix(e)
            element_energies[e] = d_e.T @ K_e @ d_e
        return element_energies, d, True

    def formulate_hamiltonian(self, element_energies):
        """
        Translates the structural QUBO into a QAOA Cost Hamiltonian (Ising).
        Matches the notation: A_e = (alpha_u + q_e*Delta_alpha) * A_old.
        """
        if self.E0 is None: self.E0 = np.sum(element_energies)
        
        # 1. Normalize quantities
        E_star = element_energies / self.E0
        v_star = (self.fem.A * self.fem.L) / self.V_bar
        C = (self.alpha_under * np.sum(v_star)) + self.S_under - 1.0
        
        # 2. Define QUBO Weights
        W = np.zeros(self.n_qubits)
        for e in range(self.n_elements):
            W[e] = v_star[e] * self.Delta_alpha
        W[-1] = self.Delta_S  # Slack variable

        # 3. Build QUBO Dictionary (Q)
        Q = {}
        # Element Diagonals: -Δα·E* + λ(W² + 2·C·W)
        for e in range(self.n_elements):
            Q[(e,)] = (-self.Delta_alpha * E_star[e]) + \
                      self.lambda_penalty * (W[e]**2 + 2 * C * W[e])
        
        # Slack Diagonal: λ(W² + 2·C·W)
        Q[(self.n_elements,)] = self.lambda_penalty * (W[-1]**2 + 2 * C * W[-1])
        
        # Off-Diagonals: 2·λ·Wi·Wj
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                Q[(i, j)] = 2 * self.lambda_penalty * W[i] * W[j]

        return self._qubo_to_ising(Q)

    def _qubo_to_ising(self, Q):
        """Converts binary QUBO (0,1) to Ising Z-operators (1,-1)."""
        pauli_list = []
        for keys, coeff in Q.items():
            if len(keys) == 1:
                # q_i -> (1 - Z_i)/2
                z_str = ["I"] * self.n_qubits
                z_str[keys[0]] = "Z"
                pauli_list.append(("".join(z_str[::-1]), -coeff/2))
            else:
                # q_i*q_j -> (1 - Z_i - Z_j + Z_i*Z_j)/4
                i, j = keys
                zz_str = ["I"] * self.n_qubits
                zz_str[i], zz_str[j] = "Z", "Z"
                pauli_list.append(("".join(zz_str[::-1]), coeff/4))
                for idx in [i, j]:
                    z_str = ["I"] * self.n_qubits
                    z_str[idx] = "Z"
                    pauli_list.append(("".join(z_str[::-1]), -coeff/4))
        return SparsePauliOp.from_list(pauli_list)

    def build_circuit(self, H, params):
        """QAOA Ansatz: alternating Cost and Mixer layers."""
        gammas = params[:self.p_layers]
        betas = params[self.p_layers:]
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits)) # Initial superposition
        
        for p in range(self.p_layers):
            # Cost Hamiltonian phase rotation
            for pauli, coeff in H.to_list():
                z_indices = [idx for idx, char in enumerate(pauli[::-1]) if char == "Z"]
                if not z_indices: continue
                angle = 2 * gammas[p] * coeff.real
                if len(z_indices) == 1: qc.rz(angle, z_indices[0])
                else: qc.rzz(angle, z_indices[0], z_indices[1])
            
            # Mixer Hamiltonian (Transverse Field)
            for i in range(self.n_qubits):
                qc.rx(2 * betas[p], i)
        return qc

    def optimize(self):
        """Main design loop using QAOA for the updater selection."""
        estimator = StatevectorEstimator() # Exact expectation value calculator
        sampler = StatevectorSampler()

        print(f"Starting QAOA Optimization ({self.n_qubits} qubits)...")
        print("Target Volume =", self.V_bar)
        self.objective_history = []
        self.volume_history = []
        self.area_history = []
        tStart = time.time()
        for it in range(self.max_iterations):
            energies, d, valid = self.get_element_energies()
            if not valid: break
            
            H_cost = self.formulate_hamiltonian(energies)

            # 1. Classical Optimization of QAOA Parameters
            def objective(params):
                qc = self.build_circuit(H_cost, params)
                return estimator.run([(qc, H_cost)]).result()[0].data.evs

            res = minimize(objective, self.optimal_params, method='SLSQP', 
                           bounds=[(0, np.pi)] * (2 * self.p_layers))
            self.optimal_params = res.x # use optimized params for next iteration

            # 2. Measurement (Sampling the best design)
            opt_qc = self.build_circuit(H_cost, self.optimal_params)
            if self.use_exact:
                # Finds the bitstring with the absolute highest amplitude. 
                # Zero statistical noise.  The result is perfectly deterministic.
                # Memory usage doubles with every qubit added ($2^n$).
                state = Statevector.from_instruction(opt_qc)
                probs = state.probabilities_dict()
                best_bitstring = max(probs, key=probs.get) 
            else:
                # Simulates many "shots" and finds the most frequent result.
                # Noisy. Results may vary slightly between runs due to sampling noise.
                # Can simulate systems up to ~30-40 qubits on classical hardware.
                opt_qc.measure_all()
                result = sampler.run([opt_qc]).result()
                counts = result[0].data.meas.get_counts()
                best_bitstring = max(counts, key=counts.get) 
            
            bits = [int(b) for b in best_bitstring[::-1]]

            # 3. Multiplicative Area Update
            alpha = np.array([self.alpha_under + bits[e] * self.Delta_alpha 
                             for e in range(self.n_elements)])
            
            self.fem.A *= alpha
            
            
            compliance = np.sum(energies)
            vol = np.sum(self.fem.A * self.fem.L) 
            volfrac = vol / self.V_bar

            self.objective_history.append(compliance)
            self.volume_history.append(vol)
            self.area_history.append(self.fem.A.copy())
            if self.verbose:
                vol = np.sum(self.fem.A * self.fem.L)
                print(f"Iter {it+1}: Compliance={np.sum(energies):.4g}, Vol={vol:.4g}")
        
            # Early stopping: quit if volume is close to desired (0.01) and compliance has stabilized (relative change < 0.05)

            if self.earlyStop and len(self.objective_history) > 1:
                vol_close = abs(volfrac - 1.0) < 0.01
                rel_comp = abs(self.objective_history[-1] - self.objective_history[-2]) / max(self.objective_history[-2], 1e-8)
                comp_stable = rel_comp < 0.005
                if vol_close and comp_stable:
                    if self.verbose:
                        print(f"Early stopping at iteration {it+1}: volume and compliance stabilized.")
                    break
        # Package results
        total_time = time.time() - tStart
        iteration = it
        results = {
            'A_final': self.fem.A,
            'area_history': self.area_history,
            'objective_history': self.objective_history,
            'volume_history': self.volume_history,
            'iterations': iteration + 1,
            'qaoa_params': self.optimal_params,
            'total_time': total_time,
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

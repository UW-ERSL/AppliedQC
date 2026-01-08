import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler

class QAOATrussOptimizerFLRS:
    def __init__(self, fem_model, V_bar, lambda_penalty=100.0, p_layers=6, max_iterations=10, verbose=True):
        self.fem = fem_model
        self.V_bar = V_bar
        self.lambda_penalty = lambda_penalty
        self.p_layers = p_layers
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Xiao et al. On-Off Encoding parameters
        self.r = np.array([0.1, 1.0])
        self.k = np.array([2.0, 4.0])
        self.k_sum = np.sum(self.k) # 6.0
        
        self.n_elements = fem_model.n_elements
        self.n_qubits = 2 * self.n_elements + 2
        
        # Restore original FLRS scales
        self.gammas, self.betas = self._generate_flrs(p_layers, gamma_max=0.6, beta_max=0.3)
        
    def _generate_flrs(self, p, gamma_max=0.6, beta_max=0.3):
        """Fixed Linear Ramp Schedule as defined in Xiao et al."""
        gammas = [(i + 1) / p * gamma_max for i in range(p)]
        betas = [(1 - i / p) * beta_max for i in range(p)]
        return gammas, betas

    def get_element_energies(self):
        d, valid = self.fem.solve()
        if not valid:
            return None, None, False
        self.fem.displacements = d
        element_energies = np.zeros(self.n_elements)
        for elem_idx in range(self.n_elements):
            d_elem = self.fem.get_element_displacements(elem_idx)
            K_elem = self.fem.get_element_stiffness_matrix(elem_idx)
            element_energies[elem_idx] = d_elem.T @ K_elem @ d_elem
        return element_energies, d, valid

    def formulate_hamiltonian(self, element_energies):
        # 1. DYNAMIC NORMALIZATION (Xiao et al. Section 3.2)
        # This prevents compliance from 'exploding' in the Hamiltonian
        E_total = np.sum(element_energies)
        e_star = element_energies / E_total
        
        # 2. VOLUME RATIO
        # Note: Use current areas to calculate the weight of the bits
        v_ratios = (self.fem.A * self.fem.L) / self.V_bar
        
        W = np.zeros(self.n_qubits)
        for e in range(self.n_elements):
            W[2*e] = v_ratios[e] * self.r[0]   # r1 = 0.1
            W[2*e+1] = v_ratios[e] * self.r[1] # r2 = 1.0
        
        # Slack normalization (Eq 27)
        W[-2] = self.k[0] / self.k_sum # 2/6
        W[-1] = self.k[1] / self.k_sum # 4/6
        
        Q = {}
        # Stiffness Term (Maximize Stiffness -> Minimize -Stiffness)
        for e in range(self.n_elements):
            Q[(2*e,)] = Q.get((2*e,), 0) - (e_star[e] * self.r[0])
            Q[(2*e+1,)] = Q.get((2*e+1,), 0) - (e_star[e] * self.r[1])
                
        # Penalty Term: lambda * (Sum(W_i * q_i) - 1)^2
        # For stability, lambda should be higher (e.g., 50-100) 
        # but ONLY if the stiffness term is normalized as above.
        target = 1.0
        for i in range(self.n_qubits):
            # Linear: lambda * (W_i^2 - 2*target*W_i)
            Q[(i,)] = Q.get((i,), 0) + self.lambda_penalty * (W[i]**2 - 2 * target * W[i])
            for j in range(i + 1, self.n_qubits):
                # Quadratic: lambda * (2*W_i*W_j)
                Q[(i, j)] = Q.get((i, j), 0) + self.lambda_penalty * (2 * W[i] * W[j])

        return self._dict_to_pauli(Q)

    def _dict_to_pauli(self, Q):
        pauli_list = []
        for keys, coeff in Q.items():
            if len(keys) == 1:
                z_str = ["I"] * self.n_qubits
                z_str[keys[0]] = "Z"
                pauli_list.append(("".join(z_str[::-1]), -coeff/2))
            else:
                i, j = keys
                zz_str = ["I"] * self.n_qubits
                zz_str[i], zz_str[j] = "Z", "Z"
                pauli_list.append(("".join(zz_str[::-1]), coeff/4))
                for idx in [i, j]:
                    z_str = ["I"] * self.n_qubits
                    z_str[idx] = "Z"
                    pauli_list.append(("".join(z_str[::-1]), -coeff/4))
        return SparsePauliOp.from_list(pauli_list)

    def build_circuit(self, hamiltonian):
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits)) 
        for p in range(self.p_layers):
            gamma = self.gammas[p]
            for pauli, coeff in hamiltonian.to_list():
                z_indices = [idx for idx, char in enumerate(pauli[::-1]) if char == "Z"]
                if not z_indices: continue
                angle = 2 * gamma * coeff.real
                if len(z_indices) == 1: qc.rz(angle, z_indices[0])
                elif len(z_indices) == 2: qc.rzz(angle, z_indices[0], z_indices[1])
            beta = self.betas[p]
            for i in range(self.n_qubits): qc.rx(2 * beta, i)
        qc.measure_all()
        return qc

    def optimize(self):
        sampler = StatevectorSampler()
        for it in range(self.max_iterations):
            energies, d, valid = self.get_element_energies()
            if not valid: break
            
            H = self.formulate_hamiltonian(energies)
            result = sampler.run([self.build_circuit(H)]).result()
            best_bitstring = max(result[0].data.meas.get_counts(), key=result[0].data.meas.get_counts().get)
            bits = [int(b) for b in best_bitstring[::-1]]
            
            updaters = []
            for e in range(self.n_elements):
                alpha_e = self.r[0] * bits[2*e] + self.r[1] * bits[2*e+1]
                # Numerical floor for FEM stability
                updaters.append(min(max(alpha_e, 0.9), 1.1))
            
            self.fem.A *= np.array(updaters)
            if self.verbose:
                vol_ratio = np.sum(self.fem.A * self.fem.L) / self.V_bar
                print(f"Iter {it+1}: Compliance={np.sum(energies):.4e}, Vol Ratio={vol_ratio:.4f}")
        return self.fem.A
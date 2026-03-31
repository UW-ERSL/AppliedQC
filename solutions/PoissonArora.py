import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator

class QFEM_2D_Encoding:
    def __init__(self, ex, ey):
        self.ex = ex
        self.ey = ey
        self.n_nodes = (ex + 1) * (ey + 1)
        self.n_qubits = int(np.ceil(np.log2(self.n_nodes)))
        self.dim = 2**self.n_qubits
        
    def get_element_stiffness(self, lx=1.0, ly=1.0, k=1.0):
        """Standard 2D bilinear quad element (4 nodes)"""
        a = k * ly / (6.0 * lx)
        b = k * lx / (6.0 * ly)
        
        ke = np.array([
            [ 2*(a+b), -2*a+b,  -a-b,    a-2*b  ],
            [-2*a+b,   2*(a+b),  a-2*b,  -a-b   ],
            [-a-b,     a-2*b,   2*(a+b), -2*a+b ],
            [ a-2*b,  -a-b,    -2*a+b,   2*(a+b)]
        ])
        return ke
    
    def embed_local_to_global(self, ke_local, node_indices):
        """
        KEY STEP from Arora et al.: 
        Embed 4×4 local matrix into N×N global matrix
        """
        K_global = np.zeros((self.dim, self.dim))
        
        # Map local 4×4 block to global positions
        for i, global_i in enumerate(node_indices):
            for j, global_j in enumerate(node_indices):
                if global_i < self.dim and global_j < self.dim:
                    K_global[global_i, global_j] = ke_local[i, j]
                    
        return K_global
    
    def assemble_global_lcu(self):
        """Arora's Q-FEM assembly"""
        ke_local = self.get_element_stiffness()
        
        # Accumulate all unitary terms
        all_coeffs = []
        all_paulis = []
        
        for ey_i in range(self.ey):
            for ex_i in range(self.ex):
                # Global node indices for this element
                node_indices = [
                    ey_i * (self.ex + 1) + ex_i,
                    ey_i * (self.ex + 1) + (ex_i + 1),
                    (ey_i + 1) * (self.ex + 1) + (ex_i + 1),
                    (ey_i + 1) * (self.ex + 1) + ex_i
                ]
                
                # Embed local 4×4 into global N×N
                K_e_global = self.embed_local_to_global(ke_local, node_indices)
                
                # Decompose this N×N matrix into Paulis
                sp_op = SparsePauliOp.from_operator(K_e_global)
                
                all_coeffs.extend(sp_op.coeffs)
                all_paulis.extend(sp_op.paulis)
        
        # Combine duplicate Pauli terms
        combined = {}
        for c, p in zip(all_coeffs, all_paulis):
            label = p.to_label()
            combined[label] = combined.get(label, 0) + c
            
        return combined

if __name__ == "__main__":
    qfem = QFEM_2D_Encoding(ex=2, ey=2)  # Smaller for testing
    lcu_data = qfem.assemble_global_lcu()
    
    alpha = sum(abs(c) for c in lcu_data.values())
    
    print(f"Elements: {qfem.ex * qfem.ey}")
    print(f"Nodes: {qfem.n_nodes}")
    print(f"Qubits needed: {qfem.n_qubits}")
    print(f"Unique Pauli terms (L): {len(lcu_data)}")
    print(f"Alpha (LCU norm): {alpha.real:.4f}")
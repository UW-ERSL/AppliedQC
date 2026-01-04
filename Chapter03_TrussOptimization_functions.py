import numpy as np
import matplotlib.pyplot as plt
import time


def truss3x3():
    # Example usage with the 3x3 truss
    nodes = np.array([
        [0.0, 0.0],   # Node 0 (bottom left) - FIXED
        [2.0, 0.0],   # Node 1 (bottom center)
        [4.0, 0.0],   # Node 2 (bottom right) - FIXED
        [0.0, 1.5],   # Node 3 (middle left)
        [2.0, 1.5],   # Node 4 (middle center)
        [4.0, 1.5],   # Node 5 (middle right)
        [0.0, 3.0],   # Node 6 (top left)
        [2.0, 3.0],   # Node 7 (top center) - LOADED
        [4.0, 3.0]    # Node 8 (top right)
    ])

    elements = [
        (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),  # Horizontal
        (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8),  # Vertical
        (0, 4), (1, 3), (1, 5), (2, 4), (3, 7), (4, 6), (4, 8), (5, 7),  # Diagonals
        (0, 8), (2, 6), (0, 7), (1, 6), (1, 8), (2, 7)   # Long diagonals
    ]

    

    # Define problem
    fixed_dofs = [0, 1, 4, 5]  # Nodes 0 and 2 fixed
    loads = np.zeros(2 * len(nodes))
    loads[2*7 + 1] = -10000  # 10 kN downward at node 7

    # Create FEM model
    fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=200e9, A=0.001, rho=7850)

    return fem_model

def truss2x2():
    # Example usage with the 2x2 truss
    nodes = np.array([
        [0.0, 0.0],   # Node 0 (bottom left) - FIXED
        [2.0, 0.0],   # Node 1 (bottom right) - FIXED
        [0.0, 2.0],   # Node 2 (top left)
        [2.0, 2.0]    # Node 3 (top right) - LOADED
    ])

    elements = [
        (0, 1), (2, 3),  # Horizontal
        (0, 2), (1, 3),  # Vertical
        (0, 3), (1, 2)   # Diagonals
    ]

   

    # Define problem
    fixed_dofs = [0, 1, 2, 3]  # Nodes 0 and 1 fixed
    loads = np.zeros(2 * len(nodes))
    loads[2*3 + 1] = -10000  # 10 kN downward at node 3

     # Create FEM model
    fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=200e9, A=0.001, rho=7850)

    return fem_model

class TrussFEM:
    def __init__(self, nodes, elements,loads, fixed_dofs, E=200e9, A=0.01, rho=7850):
        """
        Finite element model for 2D truss structures.
        
        Parameters:
        -----------
        nodes : ndarray, shape (n_nodes, 2)
            Node coordinates [x, y]
        elements : list of tuples (i, j)
            Element connectivity: each element connects nodes i and j
        E : float
            Young's modulus (Pa)
        A : float
            Cross-sectional area (m^2)
        rho : float
            Material density (kg/m^3)
        """
        self.nodes = np.array(nodes)
        self.elements = elements
        self.n_elements = len(elements)
        self.E = E
        self.loads = loads
        self.fixed_dofs = fixed_dofs
        # Allow A to be a scalar or array; convert to array of length n_elements
        if np.isscalar(A):
            self.A = np.full(self.n_elements, A)
        else:
            self.A = np.asarray(A)
            if self.A.shape[0] != self.n_elements:
                raise ValueError(" A must be a scalar or match number of elements")
        self.rho = rho
        self.n_nodes = len(nodes)
        self.n_dof = 2 * self.n_nodes
        self.lengths = self.compute_all_lengths()
        self.initial_weight = rho*np.sum(self.A * self.lengths)
    
    def compute_element_length(self, i, j): 
        """Compute length of element between nodes i and j."""
        dx = self.nodes[j, 0] - self.nodes[i, 0]
        dy = self.nodes[j, 1] - self.nodes[i, 1]
        return np.sqrt(dx**2 + dy**2)
    
    def compute_all_lengths(self):
        """Compute all lengths."""
        lengths = []
        for idx, (i, j) in enumerate(self.elements):
            lengths.append(self.compute_element_length(i, j))
        return np.array(lengths)
        
    def element_stiffness(self, i, j, area):
        """
        Compute element stiffness matrix in global coordinates.
        
        Parameters:
        -----------
        i, j : int
            Node indices
            
        Returns:
        --------
        K_elem : ndarray, shape (4, 4)
            Element stiffness matrix
        L : float
            Element length
        """
        # Element geometry
        dx = self.nodes[j, 0] - self.nodes[i, 0]
        dy = self.nodes[j, 1] - self.nodes[i, 1]
        L = np.sqrt(dx**2 + dy**2)
        
        if L < 1e-10:
            raise ValueError(f"Degenerate element: nodes {i} and {j} coincide")
        
        # Direction cosines
        c = dx / L
        s = dy / L
        
        # Element stiffness in global coordinates
        k = self.E * area / L
        K_elem = k * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        
        return K_elem
    
    def assemble_stiffness(self, area=None):
        """
        Assemble global stiffness matrix.
        
        Parameters:
        -----------
        area : array-like, shape (n_elements,)
            Cross-sectional areas for each element
            
        Returns:
        --------
        K : ndarray, shape (n_dof, n_dof)
            Global stiffness matrix
        lengths : ndarray
            Element lengths (for computing weight)
        """
        K = np.zeros((self.n_dof, self.n_dof))
        
        if area is None:
            area = self.A
        
        for idx, (i, j) in enumerate(self.elements):
            K_elem = self.element_stiffness(i, j, area[idx])
     
            # Global DOF indices for this element
            dofs = [2*i, 2*i+1, 2*j, 2*j+1]
            
            # Add element contribution to global matrix
            for a in range(4):
                for b in range(4):
                    K[dofs[a], dofs[b]] += K_elem[a, b]
        
        return K
    
    def solve(self, area = None):
        """
        Solve equilibrium equations for given design.
        Automatically excludes hanging nodes (nodes with no connected members).
        
        Validity Requirements:
        - At least one element must connect to each force application node
        - At least one element must connect to each fixed support node
        
        Parameters:
        -----------
        loads : ndarray, shape (n_dof,)
            Applied loads at each DOF
        fixed_dofs : list
            Indices of constrained DOFs
        area : array-like
            Cross-sectional areas for each element
            
        Returns:
        --------
        u : ndarray, shape (n_dof,)
            Displacement vector (hanging nodes have zero displacement)
        valid : bool
            True if solution is valid (no singularity, constraints satisfied)
        """
        
        if area is None:
            area = self.A

        loads = self.loads
        fixed_dofs = self.fixed_dofs
        # Identify connected nodes (nodes with at least one active member)
        connected_nodes = set()
        for idx, (i, j) in enumerate(self.elements):
            if area[idx] > 0:
                connected_nodes.add(i)
                connected_nodes.add(j)
        
        # Condition 1: Check that force nodes have at least one connected element
        force_nodes = set()
        for dof_idx in range(len(loads)):
            if loads[dof_idx] != 0:
                node_id = dof_idx // 2
                force_nodes.add(node_id)
        
        for force_node in force_nodes:
            if force_node not in connected_nodes:
                return np.zeros(self.n_dof), False
        
        # Condition 2: Check that each fixed node has at least one connected element
        fixed_nodes = set()
        for dof_idx in fixed_dofs:
            node_id = dof_idx // 2
            fixed_nodes.add(node_id)
        
        for fixed_node in fixed_nodes:
            if fixed_node not in connected_nodes:
                return np.zeros(self.n_dof), False
        
        # DOFs for connected nodes only
        connected_dofs = set()
        for node in connected_nodes:
            connected_dofs.add(2 * node)
            connected_dofs.add(2 * node + 1)
        
        # Free DOFs = connected DOFs minus fixed DOFs
        free_dofs = sorted(list(connected_dofs - set(fixed_dofs)))
        
        if len(free_dofs) == 0:
            # No free DOFs means structure is either empty or fully constrained
            return np.zeros(self.n_dof), False
        
        K = self.assemble_stiffness(area)
        # Extract free-free submatrix
        K_free = K[np.ix_(free_dofs, free_dofs)]
        f_free = loads[free_dofs]
        
        # Check conditioning
        if np.linalg.cond(K_free) > 1e12:
            return np.zeros(self.n_dof), False
        
        # Check for mechanism (rank deficiency)
        rank = np.linalg.matrix_rank(K_free)
        if rank < len(free_dofs):
            # Structure has mechanisms (insufficient constraints)
            return np.zeros(self.n_dof), False
        
        # Solve reduced system
        try:
            u_free = np.linalg.solve(K_free, f_free)
        except np.linalg.LinAlgError:
            return np.zeros(self.n_dof), False
        
        # Reconstruct full displacement vector
        d = np.zeros(self.n_dof)
        d[free_dofs] = u_free
        # Hanging nodes and fixed nodes remain at zero displacement
        
        return d, True
    
    def compute_compliance(self, area, displacements):
        """Compute compliance."""
        return displacements @ self.assemble_stiffness(area)[0] @ displacements
    
    def compute_stresses(self, area, displacements):
        """
        Compute member stresses from displacement solution.
        
        Parameters:
        -----------
        area : array-like
            Cross-sectional areas for each element
        d : ndarray
            Displacement vector
            
        Returns:
        --------
        stresses : ndarray
            Stress in each element (0 for inactive elements)
        """
        stresses = np.zeros(len(self.elements))
        d = displacements
        for idx, (i, j) in enumerate(self.elements):
            if area[idx] == 0:
                continue
            
            # Element geometry
            dx = self.nodes[j, 0] - self.nodes[i, 0]
            dy = self.nodes[j, 1] - self.nodes[i, 1]
            L = np.sqrt(dx**2 + dy**2)
            c, s = dx/L, dy/L
            
            # Element displacements
            d_elem = np.array([d[2*i], d[2*i+1], d[2*j], d[2*j+1]])
            
            # Axial strain
            epsilon = (1/L) * np.array([-c, -s, c, s]) @ d_elem
            
            # Stress
            stresses[idx] = self.E * epsilon
        
        return stresses
    
    def compute_elem_strain_energies(self, area, displacements):
        """
        Compute member stresses from displacement solution.
        
        Parameters:
        -----------
        area : array-like
            Cross-sectional areas for each element
        d : ndarray
            Displacement vector
            
        Returns:
        --------
        stresses : ndarray
            Stress in each element (0 for inactive elements)
        """
        strainEnergy = np.zeros(len(self.elements))
        d = displacements
        for idx, (i, j) in enumerate(self.elements):
            if area[idx] == 0:
                continue
            
            # Element geometry
            dx = self.nodes[j, 0] - self.nodes[i, 0]
            dy = self.nodes[j, 1] - self.nodes[i, 1]
            L = np.sqrt(dx**2 + dy**2)
            c, s = dx/L, dy/L
            
            # Element displacements
            d_elem = np.array([d[2*i], d[2*i+1], d[2*j], d[2*j+1]])
            
            # Axial strain
            epsilon = (1/L) * np.array([-c, -s, c, s]) @ d_elem
            
            # Stress
            stress = self.E * epsilon
            strainEnergy[idx] = 0.5 * stress * epsilon * area[idx] * L
        
        return strainEnergy
    
    def compute_weight(self, area):
        """Compute total weight of design."""
        
        return self.rho * np.sum(area * self.lengths)


    def evaluate_design(self, area = None,desiredWeightFraction = 1,
                   d_hat=np.inf, sigma_hat=np.inf):
        """
        Fully evaluate a design: solve FEM and check constraints.
        """
        if area is None:
            area = self.A
        d, valid = self.solve(area)
        
        if not valid:
            return {
                'weight': np.inf,
                'max_disp': np.inf,
                'max_stress': np.inf,
                'feasible': False,
                'compliance': np.inf
            }
        
        stresses = self.compute_stresses(area, d)
        weight = self.compute_weight(area)
        max_disp = np.max(np.abs(d))
        max_stress = np.max(np.abs(stresses))
        compliance = self.loads @ d
        
        desiredWeight = desiredWeightFraction*self.initial_weight   
        # Count connected nodes (nodes with at least one active member)
        connected_nodes = set()
        for idx, (i, j) in enumerate(self.elements):
            if area[idx] > 0:
                connected_nodes.add(i)
                connected_nodes.add(j)
        n_connected = len(connected_nodes)
        
        # Minimum members for connected structure: 2n - 3 (2D truss)
        min_members_required = max(1, 2 * n_connected - 3)
        
        feasible = (max_disp <= d_hat and 
                max_stress <= sigma_hat and
                np.sum(area > 0) >= min_members_required and
                weight <= desiredWeight)
        
        metrics = {
            'weight': weight,
            'max_disp': max_disp,
            'max_stress': max_stress,
            'feasible': feasible,
            'compliance': compliance
        }
         
        
        return metrics

    def print_metrics(self, metrics):
        """
        Print performance metrics in a formatted way.
        
        Parameters:
        -----------
        metrics : dict
            Performance metrics dictionary
        """
        print(f"  Weight: {metrics['weight']:.2f} kg")
        print(f"  Max displacement: {metrics['max_disp']*1000:.4f} mm")
        print(f"  Max stress: {metrics['max_stress']/1e6:.2f} MPa")
        print(f"  Compliance: {metrics['compliance']:.2f} J")
        print(f"  Feasible: {metrics['feasible']}")

    def plot_truss(self, design=None, 
               displacements=None,
               show_nodes=True, show_labels=False,
               title="Truss Structure", figsize=(12, 8),
               save_path=None):
        """
        Visualize truss structure with optional deformed shape.
        
        Parameters:
        -----------
        design : array-like, optional
            Binary array indicating active elements. If None, plot all elements.
        displacements : ndarray, optional
            Displacement vector (for plotting deformed shape)
        show_nodes : bool
            Whether to show node markers
        show_labels : bool
            Whether to show node number labels
        title : str
            Plot title
        figsize : tuple
            Figure size
        save_path : str, optional
            If provided, save figure to this path
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        scale_factor = 0.1/abs(displacements).max() if displacements is not None else 1.0
        # Determine which elements to plot
        if design is None:
            active_elements = np.ones(len(self.elements), dtype=bool)
        else:
            active_elements = np.array(design, dtype=bool)
        
        # Plot undeformed structure
        for idx, (i, j) in enumerate(self.elements):
            if active_elements[idx]:
                ax.plot([self.nodes[i, 0], self.nodes[j, 0]], 
                    [self.nodes[i, 1], self.nodes[j, 1]], 
                    'k-', linewidth=2, alpha=0.7, zorder=1)
            else:
                ax.plot([self.nodes[i, 0], self.nodes[j, 0]], 
                    [self.nodes[i, 1], self.nodes[j, 1]], 
                    color='lightgray', linewidth=0.5, alpha=0.3, zorder=1)
        
        # Plot deformed structure if displacements provided
        if displacements is not None:
            deformed_nodes = self.nodes.copy()
            for i in range(self.n_nodes):
                deformed_nodes[i, 0] += scale_factor * displacements[2*i]
                deformed_nodes[i, 1] += scale_factor * displacements[2*i+1]
            
            for idx, (i, j) in enumerate(self.elements):
                if active_elements[idx]:
                    ax.plot([deformed_nodes[i, 0], deformed_nodes[j, 0]], 
                        [deformed_nodes[i, 1], deformed_nodes[j, 1]], 
                        'b--', linewidth=1.5, alpha=0.7, zorder=2,
                        label='Deformed' if idx == 0 else '')
            
            # Plot deformed nodes
            if show_nodes:
                ax.plot(deformed_nodes[:, 0], deformed_nodes[:, 1], 
                    'bo', markersize=6, zorder=5, alpha=0.7)
        
        # Plot undeformed nodes
        if show_nodes:
            ax.plot(self.nodes[:, 0], self.nodes[:, 1], 
                'o', color='lightgray', markersize=8, zorder=4)
        
        # Mark fixed supports
        if self.fixed_dofs is not None:
            fixed_nodes = set()
            for dof in self.fixed_dofs:
                node_idx = dof // 2
                fixed_nodes.add(node_idx)
            
            fixed_nodes = list(fixed_nodes)
            if fixed_nodes:
                ax.plot(self.nodes[fixed_nodes, 0], 
                    self.nodes[fixed_nodes, 1], 
                    'ks', markersize=12, zorder=6, 
                    markerfacecolor='black')
        
        # Draw load arrows
        if self.loads is not None:
            arrow_scale = 0.5  # Arrow length relative to structure size
            max_load = np.max(np.abs(self.loads[self.loads != 0])) if np.any(self.loads != 0) else 1.0
            
            load_drawn = False
            for i in range(self.n_nodes):
                fx = self.loads[2*i]
                fy = self.loads[2*i+1]
                
                if abs(fx) > 1e-6 or abs(fy) > 1e-6:
                    # Compute arrow length (proportional to load magnitude)
                    magnitude = np.sqrt(fx**2 + fy**2)
                    arrow_len = arrow_scale * (magnitude / max_load)
                    
                    # Normalized force direction
                    fx_norm = fx / magnitude
                    fy_norm = fy / magnitude
                    
                    # Arrow starts away from node (opposite to force direction)
                    # and points toward node (in force direction)
                    start_x = self.nodes[i, 0] - arrow_len * fx_norm
                    start_y = self.nodes[i, 1] - arrow_len * fy_norm
                    
                    # Arrow displacement (in direction of force)
                    dx = arrow_len * fx_norm
                    dy = arrow_len * fy_norm
                    
                    ax.arrow(start_x, start_y, dx, dy,
                            head_width=0.15, head_length=0.1, 
                            fc='red', ec='red', linewidth=2.5, zorder=7,
                            )
                    load_drawn = True
                    
                    # Add load magnitude label (at start of arrow)
                    load_kN = magnitude / 1000
                    ax.text(start_x, start_y, 
                        f'{load_kN:.1f} kN',
                        color='red', fontsize=11, fontweight='bold',
                        ha='center', va='bottom' if fy < 0 else 'top',
                        bbox=dict(boxstyle='round,pad=0.4',
                                    facecolor='white', edgecolor='red', alpha=0.9))
        
        # Add node labels
        if show_labels:
            for i, (x, y) in enumerate(self.nodes):
                ax.text(x, y+0.15, f'{i}', ha='center', va='bottom',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                    facecolor='white', alpha=0.7))
        
        ax.set_xlabel('x (m)', fontsize=12)
        ax.set_ylabel('y (m)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
      
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()

    def random_search(self,M=10000, desiredWeightFraction=1.0, seed =None):
        """
        
        
        Parameters:
        -----------
        M : int
            Number of random samples to evaluate (default 1000)
        desiredWeightFraction : float
            Desired fraction of the maximum weight (default 1.0)
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        weights, n_members, compliances, designs_evaluated : lists
            Design space data for feasible designs
        """
        import time
        
        if seed is not None:
            np.random.seed(seed)
        
        weights = []
        n_members = []
        compliances = []
        designs_evaluated = []
        
        N = self.n_elements
        print(f"\nSampling {M:,} random designs from 2^{N} = {2**N:,} possible designs")
        print(f"Sampling rate: {100*M/2**N:.4f}%\n")
        
        start_time = time.time()
        
        # Generate M random designs with varying member counts
        # Use different probabilities to explore design space better
        probabilities = [0.3, 0.5, 0.7]  # Sparse, medium, dense
        samples_per_prob = M // len(probabilities)
        
        random_designs = []
        for p in probabilities:
            for _ in range(samples_per_prob):
                design = (np.random.rand(N) < p).astype(int)
                random_designs.append(design)
        
        # Add remaining samples with uniform probability
        for _ in range(M - len(random_designs)):
            design = np.random.randint(0, 2, N)
            random_designs.append(design)
        
        # Evaluate all random designs with progress updates
        print_interval = max(1, M // 10)  # Print every 10%
        
        for idx, design in enumerate(random_designs):
            area = design*self.A  # Scale areas 
            metrics = self.evaluate_design(area, desiredWeightFraction=desiredWeightFraction)
            
            if metrics['feasible']:
                weights.append(metrics['weight'])
                n_members.append(np.sum(design))
                compliances.append(metrics['compliance'])
                designs_evaluated.append(design.copy())
            
            # Print progress
            if (idx + 1) % print_interval == 0 or idx == M - 1:
                elapsed = time.time() - start_time
                percent = 100 * (idx + 1) / M
                rate = (idx + 1) / elapsed
                remaining = (M - idx - 1) / rate if rate > 0 else 0
                
                print(f"Progress: {percent:5.1f}% ({idx+1:,}/{M:,}) | "
                    f"Elapsed: {elapsed:.1f}s | "
                    f"Remaining: ~{remaining:.1f}s | "
                    f"Feasible: {len(weights)}")
        
        elapsed = time.time() - start_time
        
        print(f"\nCompleted in {elapsed:.2f} seconds ({elapsed/M*1000:.2f} ms per design)")
        print(f"Found {len(weights)} feasible designs out of {M:,} sampled")
        print(f"Feasibility rate: {100*len(weights)/M:.2f}%")
        
        if len(weights) == 0:
            print("\nWARNING: No feasible designs found! Try:")
            print("  - Increasing M (more samples)")
            print("  - Relaxing constraints (u_hat, sigma_hat)")
            print("  - Checking problem formulation")
            return [], [], [], []
        
        # Print statistics
        print(f"\n{'='*70}")
        print("Design Space Statistics:")
        print(f"{'='*70}")
        best_idx = weights.index(min(weights))
    
        
        print(f"Best design:")
        print(f"  Weight: {weights[best_idx]:.2f} kg")
        print(f"  Members: {n_members[best_idx]}")
        print(f"  Compliance: {compliances[best_idx]:.2f} J")
        print(f"  Active members: {np.where(designs_evaluated[best_idx])[0].tolist()}")

        best_design = designs_evaluated[weights.index(min(weights))]
        return best_design




import numpy as np
import matplotlib.pyplot as plt
import time

class TrussFEM:
    def __init__(self, nodes, elements, E=200e9, A=0.01, rho=7850):
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
        self.E = E
        self.A = A
        self.rho = rho
        self.n_nodes = len(nodes)
        self.n_dof = 2 * self.n_nodes
        
    def element_stiffness(self, i, j):
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
        k = self.E * self.A / L
        K_elem = k * np.array([
            [ c*c,  c*s, -c*c, -c*s],
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        
        return K_elem, L
    
    def assemble_stiffness(self, design):
        """
        Assemble global stiffness matrix.
        
        Parameters:
        -----------
        design : array-like, shape (n_elements,)
            Binary array: 1 if element is active, 0 otherwise
            
        Returns:
        --------
        K : ndarray, shape (n_dof, n_dof)
            Global stiffness matrix
        lengths : ndarray
            Element lengths (for computing weight)
        """
        K = np.zeros((self.n_dof, self.n_dof))
        lengths = np.zeros(len(self.elements))
        
        for idx, (i, j) in enumerate(self.elements):
            if design[idx] == 0:
                continue  # Element not in design
            
            K_elem, L = self.element_stiffness(i, j)
            lengths[idx] = L
            
            # Global DOF indices for this element
            dofs = [2*i, 2*i+1, 2*j, 2*j+1]
            
            # Add element contribution to global matrix
            for a in range(4):
                for b in range(4):
                    K[dofs[a], dofs[b]] += K_elem[a, b]
        
        return K, lengths
    
    def solve(self, design, loads, fixed_dofs):
        """
        Solve equilibrium equations for given design.
        Automatically excludes hanging nodes (nodes with no connected members).
        
        Validity Requirements:
        - At least one element must connect to each force application node
        - At least one element must connect to each fixed support node
        
        Parameters:
        -----------
        design : array-like
            Binary array indicating active elements
        loads : ndarray, shape (n_dof,)
            Applied loads at each DOF
        fixed_dofs : list
            Indices of constrained DOFs
            
        Returns:
        --------
        u : ndarray, shape (n_dof,)
            Displacement vector (hanging nodes have zero displacement)
        valid : bool
            True if solution is valid (no singularity, constraints satisfied)
        """
        
        # Identify connected nodes (nodes with at least one active member)
        connected_nodes = set()
        for idx, (i, j) in enumerate(self.elements):
            if design[idx] == 1:
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
        
        K, lengths = self.assemble_stiffness(design)
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
        u = np.zeros(self.n_dof)
        u[free_dofs] = u_free
        # Hanging nodes and fixed nodes remain at zero displacement
        
        return u, True
    def compute_stresses(self, design, u):
        """
        Compute member stresses from displacement solution.
        
        Parameters:
        -----------
        design : array-like
            Binary array indicating active elements
        u : ndarray
            Displacement vector
            
        Returns:
        --------
        stresses : ndarray
            Stress in each element (0 for inactive elements)
        """
        stresses = np.zeros(len(self.elements))
        
        for idx, (i, j) in enumerate(self.elements):
            if design[idx] == 0:
                continue
            
            # Element geometry
            dx = self.nodes[j, 0] - self.nodes[i, 0]
            dy = self.nodes[j, 1] - self.nodes[i, 1]
            L = np.sqrt(dx**2 + dy**2)
            c, s = dx/L, dy/L
            
            # Element displacements
            u_elem = np.array([u[2*i], u[2*i+1], u[2*j], u[2*j+1]])
            
            # Axial strain
            epsilon = (1/L) * np.array([-c, -s, c, s]) @ u_elem
            
            # Stress
            stresses[idx] = self.E * epsilon
        
        return stresses
    
    def compute_weight(self, design):
        """Compute total weight of design."""
        _, lengths = self.assemble_stiffness(design)
        return self.rho * self.A * np.sum(design * lengths)
    
    def evaluate_design(self, design, loads, fixed_dofs, 
                   u_hat=0.01, sigma_hat=250e6):
        """
        Fully evaluate a design: solve FEM and check constraints.
        """
        u, valid = self.solve(design, loads, fixed_dofs)
        
        if not valid:
            return {
                'weight': np.inf,
                'max_disp': np.inf,
                'max_stress': np.inf,
                'feasible': False,
                'compliance': np.inf
            }
        
        stresses = self.compute_stresses(design, u)
        weight = self.compute_weight(design)
        max_disp = np.max(np.abs(u))
        max_stress = np.max(np.abs(stresses))
        compliance = loads @ u
        
        # Count connected nodes (nodes with at least one active member)
        connected_nodes = set()
        for idx, (i, j) in enumerate(self.elements):
            if design[idx] == 1:
                connected_nodes.add(i)
                connected_nodes.add(j)
        n_connected = len(connected_nodes)
        
        # Minimum members for connected structure: 2n - 3 (2D truss)
        min_members_required = max(1, 2 * n_connected - 3)
        
        feasible = (max_disp <= u_hat and 
                max_stress <= sigma_hat and
                np.sum(design) >= min_members_required)
        
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

    def plot_truss(self, design=None, loads=None, fixed_dofs=None, 
               displacements=None, scale_factor=10, 
               show_nodes=True, show_labels=False,
               title="Truss Structure", figsize=(12, 8),
               save_path=None):
        """
        Visualize truss structure with optional deformed shape.
        
        Parameters:
        -----------
        design : array-like, optional
            Binary array indicating active elements. If None, plot all elements.
        loads : ndarray, optional
            Load vector (for drawing load arrows)
        fixed_dofs : list, optional
            Fixed DOF indices (for marking supports)
        displacements : ndarray, optional
            Displacement vector (for plotting deformed shape)
        scale_factor : float
            Magnification factor for deformed shape visualization
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
        if fixed_dofs is not None:
            fixed_nodes = set()
            for dof in fixed_dofs:
                node_idx = dof // 2
                fixed_nodes.add(node_idx)
            
            fixed_nodes = list(fixed_nodes)
            if fixed_nodes:
                ax.plot(self.nodes[fixed_nodes, 0], 
                    self.nodes[fixed_nodes, 1], 
                    'ks', markersize=12, zorder=6, 
                    markerfacecolor='black')
        
        # Draw load arrows
        if loads is not None:
            arrow_scale = 0.5  # Arrow length relative to structure size
            max_load = np.max(np.abs(loads[loads != 0])) if np.any(loads != 0) else 1.0
            
            load_drawn = False
            for i in range(self.n_nodes):
                fx = loads[2*i]
                fy = loads[2*i+1]
                
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

    


def exhaustive_search(fem_model, loads, fixed_dofs, N):
    """
    Find optimal design by exhaustive enumeration.
    
    Parameters:
    -----------
    fem_model : TrussFEM
        FEM model
    loads, fixed_dofs : as before
    N : int
        Number of potential members
        
    Returns:
    --------
    best_design : ndarray
        Optimal design
    best_metrics : dict
        Performance metrics of optimal design
    """
    best_weight = np.inf
    best_design = None
    best_metrics = None
    
    n_designs = 2**N
    print(f"Enumerating {n_designs} designs...")
    
    start_time = time.time()
    
    for i in range(n_designs):
        # Convert integer i to binary design vector
        design = np.array([int(b) for b in format(i, f'0{N}b')])
        
        # Evaluate design
        metrics = fem_model.evaluate_design(design, loads, fixed_dofs)
        
        # Update best if feasible and lighter
        if metrics['feasible'] and metrics['weight'] < best_weight:
            best_weight = metrics['weight']
            best_design = design.copy()
            best_metrics = metrics
    
    elapsed = time.time() - start_time
    
    print(f"Exhaustive search completed in {elapsed:.2f} seconds")
    print(f"Evaluated {n_designs} designs")
    print(f"Best weight: {best_weight:.2f}")
    
    return best_design, best_metrics

def search_design_space(fem_model, loads, fixed_dofs, N, M=10000, seed=None):
    """
    
    
    Parameters:
    -----------
    fem_model : TrussFEM
        Truss finite element model
    loads : ndarray
        Load vector
    fixed_dofs : list
        Fixed DOF indices
    N : int
        Number of potential members
    M : int
        Number of random samples to evaluate (default 1000)
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
        metrics = fem_model.evaluate_design(design, loads, fixed_dofs)
        
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


def simulated_annealing(fem_model, loads, fixed_dofs, N,
                       T_init=1000, T_min=0.1, alpha=0.95, 
                       steps_per_temp=100):
    """
    Simulated annealing for truss optimization.
    
    Parameters:
    -----------
    T_init : float
        Initial temperature
    T_min : float
        Minimum temperature (stopping criterion)
    alpha : float
        Cooling rate (T_new = alpha * T_old)
    steps_per_temp : int
        Number of iterations per temperature
        
    Returns:
    --------
    best_design : ndarray
        Best design found
    history : list
        Evolution history for plotting
    """
    # Initialize with random design
    current = np.random.randint(0, 2, N)
    current_metrics = fem_model.evaluate_design(current, loads, fixed_dofs)
    current_fitness = (current_metrics['weight'] if current_metrics['feasible']
                      else current_metrics['weight'] + 1e6)
    
    best = current.copy()
    best_fitness = current_fitness
    
    T = T_init
    history = []
    iteration = 0
    
    print(f"Starting SA: T_init={T_init}, T_min={T_min}, alpha={alpha}")
    
    while T > T_min:
        for _ in range(steps_per_temp):
            # Generate neighbor: flip random bit
            neighbor = current.copy()
            flip_idx = np.random.randint(0, N)
            neighbor[flip_idx] = 1 - neighbor[flip_idx]
            
            # Evaluate neighbor
            neighbor_metrics = fem_model.evaluate_design(neighbor, loads, 
                                                        fixed_dofs)
            neighbor_fitness = (neighbor_metrics['weight'] 
                              if neighbor_metrics['feasible']
                              else neighbor_metrics['weight'] + 1e6)
            
            # Acceptance criterion
            delta = neighbor_fitness - current_fitness
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                # Accept neighbor
                current = neighbor
                current_fitness = neighbor_fitness
                
                # Update best
                if current_fitness < best_fitness:
                    best = current.copy()
                    best_fitness = current_fitness
            
            history.append({
                'iteration': iteration,
                'temperature': T,
                'current_fitness': current_fitness,
                'best_fitness': best_fitness
            })
            iteration += 1
        
        # Cool down
        T *= alpha
        
        if iteration % 10000 == 0:
            print(f"Iteration {iteration}: T={T:.2f}, "
                  f"Best fitness={best_fitness:.2f}")
    
    print(f"\nSA completed after {iteration} iterations")
    print(f"Best fitness: {best_fitness:.2f}")
    
    return best

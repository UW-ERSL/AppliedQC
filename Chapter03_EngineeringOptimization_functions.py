import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.optimize import minimize



class TrussFEM:
    def __init__(self, nodes, elements,loads, fixed_dofs, E=200e9, A=0.0005):
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
        """
        self.nodes = np.array(nodes)
        self.n_dofs = 2 * len(nodes)
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
        self.n_nodes = len(nodes)
        self.n_dof = 2 * self.n_nodes
        self.L = self.compute_all_lengths()
        self.Te = self.compute_Te()
        self.initialArea = self.A.copy()
        self.initial_volume = np.sum(self.initialArea * self.L)
        self.displacements = None
    

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
    
    
    def set_area(self, A):
        """Set cross-sectional areas."""
        if np.isscalar(A):
            self.A = np.full(self.n_elements, A)
        else:
            self.A = np.asarray(A)
            if self.A.shape[0] != self.n_elements:
                raise ValueError("A must be a scalar or match number of elements")
            

    def get_element_displacements(self, bar_idx):
        """
        Get element displacement vector for a given element.
        
        Parameters:
        -----------
        bar_idx : int
            Element index
        d : ndarray, shape (n_dof,)
            Global displacement vector
            
        Returns:
        --------
        d_elem : ndarray, shape (4,)
            Element displacement vector [u_i, v_i, u_j, v_j]
        """
        i, j = self.elements[bar_idx]
        d_elem = np.array([
            self.displacements[2*i],
            self.displacements[2*i + 1],
            self.displacements[2*j],
            self.displacements[2*j + 1]
        ])
        return d_elem
    
    def compute_Te(self):
        """
        Compute transformation matrices Te for all elements.
        Each Te is a 4x4 matrix for mapping local to global coordinates.

        Returns:
        --------
        Te_all : ndarray, shape (n_elements, 4, 4)
            Array of transformation matrices for each element.
        """
        Te_all = np.zeros((self.n_elements, 4, 4))
        for elem in range(self.n_elements):
            i, j = self.elements[elem]
            dx = self.nodes[j, 0] - self.nodes[i, 0]
            dy = self.nodes[j, 1] - self.nodes[i, 1]
            L = self.L[elem]
            c = dx / L
            s = dy / L
            Te = np.array([
                [ c*c,  c*s, -c*c, -c*s],
                [ c*s,  s*s, -c*s, -s*s],
                [-c*c, -c*s,  c*c,  c*s],
                [-c*s, -s*s,  c*s,  s*s]
            ])
            Te_all[elem] = Te
        return Te_all
    
    def get_element_stiffness_matrix(self, elem):
        """
        Compute element stiffness matrix in global coordinates.
        
        Parameters:
        -----------
        elem : int
            Element index
            
        Returns:
        --------
        K_elem : ndarray, shape (4, 4)
            Element stiffness matrix
        L : float
            Element length
        """
        # Use precomputed transformation matrix Te
        Te = self.Te[elem]
        k = self.E * self.A[elem] / self.L[elem]
        K_elem = k * Te
        return K_elem
    
    def assemble_stiffness(self):
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
            Element lengths (for computing volume)
        """
        K = np.zeros((self.n_dof, self.n_dof))
        for idx, (i, j) in enumerate(self.elements):
            K_elem = self.get_element_stiffness_matrix(idx)
     
            # Global DOF indices for this element
            dofs = [2*i, 2*i+1, 2*j, 2*j+1]
            
            # Vectorized element contribution to global matrix
            K[np.ix_(dofs, dofs)] += K_elem
        
        return K
    
    def solve(self):
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
        
        K = self.assemble_stiffness()
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
        self.displacements = d
        return d, True
    
    def compute_compliance(self, displacements):
        """Compute compliance."""
        return displacements @ self.assemble_stiffness()[0] @ displacements
    
    def compute_stresses(self, displacements):
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
        area = self.A
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
    
    
    def compute_volume(self):
        """Compute total volume of design."""
        return np.sum(self.A * self.L)


    def optimize_areas(self, volume_fraction=0.5, xi_min=0.01, xi_max=2):
        """
        Optimizes element cross-sectional areas to minimize compliance.
        
        Parameters:
        -----------
        volume_fraction : float
            Allowed volume as a fraction of the initial volume (0 to 1).
        xi_min : float
            Lower bound for relative area to avoid numerical singularities.
        xi_max : float, optional
            Upper bound for relative area. Defaults to 1 (initial area).
        """
  
        
        # Target volume limit

        n_elems = self.n_elements
        A0 = self.initialArea
        print(f"Initial volume: {self.initial_volume:.4f} m^3")
       
        # 1. Define Objective Function (Compliance)
        def objective(xi):
            A = xi*A0
            self.set_area(A)
            u, valid = self.solve()
            if not valid:
                return 1e20  # Return large penalty if system is unstable
            
            # Compliance c = f^T * u
            compliance = self.loads @ u
            
            # Sensitivity: dc/dxi_e = -A0 * E * L_e * epsilon_e^2
            grad = np.zeros(n_elems)
            for elem in range(n_elems):
                i, j = self.elements[elem]
                
                # Element geometry
                dx = self.nodes[j, 0] - self.nodes[i, 0]
                dy = self.nodes[j, 1] - self.nodes[i, 1]
                L = self.L[elem]  # Already computed
                c, s = dx/L, dy/L
                
                # Element displacements
                u_elem = self.get_element_displacements(elem)
                
                # Axial strain
                epsilon = (1/L) * np.array([-c, -s, c, s]) @ u_elem
                
                # Gradient
                grad[elem] = -A0[elem] * self.E * L * epsilon**2
                    
            return compliance, grad

        # 3. Define Constraints (Volume)
        def volume_constraint(xi):
            # Must be >= 0 for SLSQP
            volume = np.sum(xi * A0 * self.L)
            return volume_fraction - volume/self.initial_volume

        def volume_gradient(xi):
            return -A0 * self.L / self.initial_volume

        # Constraint dictionary
        cons = {'type': 'ineq', 'fun': volume_constraint, 'jac': volume_gradient}
        
        # 4. Bounds for A (to prevent elements from disappearing entirely)
        bounds = [(xi_min, xi_max) for _ in range(n_elems)]
        
        # 5. Run Optimization
        print(f"Starting optimization (Target Volume: {volume_fraction:.4f} fraction)...")
        res = minimize(
            fun=objective,
            x0=np.ones(n_elems),
            method='SLSQP',
            jac=True,
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-9, 'disp': False, 'maxiter': 100}
        )
        
        if res.success:
            self.set_area(res.x * self.initialArea)
            print(" Optimization successful.")
            print(f"Final volume: {self.compute_volume():.3g} m^3")
        else:
            print(f"Optimization failed: {res.message}")
            
        return res

    def evaluate_design(self, desiredVolumeFraction = 1,
                   d_hat=np.inf, sigma_hat=np.inf):
        """
        Fully evaluate a design: solve FEM and check constraints.
        """
        d, valid = self.solve()
        
        if not valid:
            return {
                'volume': np.inf,
                'max_disp': np.inf,
                'max_stress': np.inf,
                'feasible': False,
                'compliance': np.inf
            }
        
        stresses = self.compute_stresses(d)
        volume = self.compute_volume()
        max_disp = np.max(np.abs(d))
        max_stress = np.max(np.abs(stresses))
        compliance = self.loads @ d
        
        desiredVolume = desiredVolumeFraction*self.initial_volume   
        # Count connected nodes (nodes with at least one active member)
        connected_nodes = set()
        for idx, (i, j) in enumerate(self.elements):
            if self.A[idx] > 0:
                connected_nodes.add(i)
                connected_nodes.add(j)
        n_connected = len(connected_nodes)
        
        # Minimum members for connected structure: 2n - 3 (2D truss)
        min_members_required = max(1, 2 * n_connected - 3)
        feasible = ( np.sum(self.A > 0) >= min_members_required and
                volume <= desiredVolume) # can add more constraints here
        
        metrics = {
            'volume': volume,
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
        print(f"  Volume: {metrics['volume']:.3g} m^3")
        print(f"  Max displacement: {metrics['max_disp']:.4g} m")
        print(f"  Max stress: {metrics['max_stress']:.3g} Pa")
        print(f"  Compliance: {metrics['compliance']:.3g} J")
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
        
        scale_factor = 0.1/(abs(displacements).max()+1e-12) if displacements is not None else 1.0
        # Determine which elements to plot
        if design is None:
            active_elements = np.ones(len(self.elements), dtype=bool)
        else:
            active_elements = np.array(design, dtype=bool)
        
        A = self.A
        # Plot undeformed structure with thickness proportional to A
        ref_A = np.mean(A)
        for idx, (i, j) in enumerate(self.elements):
            lw = 0.1 + 5.0 * (A[idx] / ref_A) if active_elements[idx] else 0.5
            color = 'k' if active_elements[idx] else 'lightgray'
            alpha = 0.7 if active_elements[idx] else 0.3
            ax.plot([self.nodes[i, 0], self.nodes[j, 0]], 
                [self.nodes[i, 1], self.nodes[j, 1]], 
                color=color, linewidth=lw, alpha=alpha, zorder=1)
            # Use a different font for node labels
            if show_labels:
                ax.text(self.nodes[i, 0], self.nodes[i, 1], str(i), color='blue', fontsize=16,
                        ha='center', va='center', fontname='Comic Sans MS', zorder=10)
                ax.text(self.nodes[j, 0], self.nodes[j, 1], str(j), color='blue', fontsize=16,
                        ha='center', va='center', fontname='Comic Sans MS', zorder=10)
            # Plot element number at 75% along the element, slightly above the member
            frac = 0.75
            x_pos = self.nodes[i, 0] + frac * (self.nodes[j, 0] - self.nodes[i, 0])
            y_pos = self.nodes[i, 1] + frac * (self.nodes[j, 1] - self.nodes[i, 1])
            # Offset perpendicular to the element
            dx = self.nodes[j, 0] - self.nodes[i, 0]
            dy = self.nodes[j, 1] - self.nodes[i, 1]
            length = np.hypot(dx, dy)
            if length > 0:
                # Perpendicular direction (normalized)
                perp_x = -dy / length
                perp_y = dx / length
                offset = 0.08  # Adjust as needed
                x_pos += offset * perp_x
                y_pos += offset * perp_y
            if (show_labels):
                ax.text(x_pos, y_pos, str(idx), color='red', fontsize=20, ha='center', va='center', alpha=0.8)
        
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
            if show_labels:
                # Show node numbers next to nodes with shaded background
                for i, (x, y) in enumerate(self.nodes):
                    ax.text(
                        x + 0.08, y + 0.08, str(i),
                        color='black', fontsize=20, ha='center', va='bottom', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', edgecolor='none', alpha=0.7)
                    )
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
            arrow_scale = 0.1  # Arrow length relative to structure size
            max_load = np.max(np.abs(self.loads[self.loads != 0])) if np.any(self.loads != 0) else 1.0
            
           
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
                    start_x = self.nodes[i, 0] 
                    start_y = self.nodes[i, 1] 
                    
                    # Arrow displacement (in direction of force)
                    dx = arrow_len * fx_norm
                    dy = arrow_len * fy_norm
                    
                    ax.arrow(start_x, start_y, dx, dy,
                            head_width=0.05, head_length=0.03, 
                            fc='red', ec='red', linewidth=2.5, zorder=7,
                            )
                    
                    # Add load magnitude label (at end of arrow)
                    load_kN = magnitude / 1000
                    end_x = start_x + dx
                    end_y = start_y + dy
                    ax.text(end_x, end_y, 
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

class PlaneStressFEM:
    """
    2D Plane Stress FEM for rectangular grid mesh
    
    Coordinate system:
        y ^
          |
          +----> x
    
    Element numbering (4-node quad):
        3---2
        |   |
        0---1
    """
    
    def __init__(self, nx, ny, lx=1.0, ly=1.0, E=200e9, nu=0.3, t=0.1):
        """
        Parameters:
        -----------
        nx, ny : int
            Number of elements in x and y directions
        lx, ly : float
            Domain dimensions (length in x and y)
        E : float
            Young's modulus
        nu : float
            Poisson's ratio
        t : float
            Thickness (for plane stress)
        """
        self.nelx = nx
        self.nely = ny
        self.lx = lx
        self.ly = ly
        self.E = E
        self.nu = nu
        self.t = t
        
        # Derived quantities
        self.n_elements = nx * ny
        self.n_nodes_x = nx + 1
        self.n_nodes_y = ny + 1
        self.n_nodes = self.n_nodes_x * self.n_nodes_y
        self.n_dofs = 2 * self.n_nodes
        
        # Element dimensions (all elements identical)
        self.elem_width = lx / nx
        self.elem_height = ly / ny
        
        # Node coordinates
        self.node_coords = self._generate_node_coordinates()
        
        # Element connectivity (which nodes form each element)
        self.connectivity = self._generate_connectivity()
        
        # Element stiffness matrix (identical for all elements)
        self.Ke = self._compute_element_stiffness()
        
        # Design variables (element densities)
        self.xi = np.ones(self.n_elements)  # Start with full material
        
        # SIMP penalization
        self.penal = 3.0  # Penalty parameter for SIMP
        self.xi_min = 1e-9  # Minimum to avoid singularity
        
        # Boundary conditions
        self.fixed_dofs = []
        self.forces = np.zeros(self.n_dofs)
        
        # Solution
        self.displacements = None  # Global displacement vector
        self.compliance = None
        self.title = "Plane Stress FEM Model"
    
    def _generate_node_coordinates(self):
        """Generate coordinates for all nodes"""
        coords = np.zeros((self.n_nodes, 2))
        
        for i in range(self.n_nodes_y):
            for j in range(self.n_nodes_x):
                node_id = i * self.n_nodes_x + j
                coords[node_id, 0] = j * self.elem_width
                coords[node_id, 1] = i * self.elem_height
        
        return coords
    
    def _generate_connectivity(self):
        """
        Generate element connectivity matrix
        
        Returns:
        --------
        connectivity : array of shape (n_elements, 4)
            connectivity[e] = [n0, n1, n2, n3] are the 4 node IDs for element e
            
        Node ordering for each element:
            3---2
            |   |
            0---1
        """
        connectivity = np.zeros((self.n_elements, 4), dtype=int)
        
        for ely in range(self.nely):
            for elx in range(self.nelx):
                elem_id = ely * self.nelx + elx
                
                # Bottom-left node
                n0 = ely * self.n_nodes_x + elx
                n1 = n0 + 1
                n2 = n0 + self.n_nodes_x + 1
                n3 = n0 + self.n_nodes_x
                
                connectivity[elem_id] = [n0, n1, n2, n3]
        
        return connectivity
    
    def _compute_element_stiffness(self):
        """
        Compute element stiffness matrix for 4-node bilinear quad
        Uses 2×2 Gauss quadrature
        
        Returns:
        --------
        Ke : array of shape (8, 8)
            Element stiffness matrix in global coordinates
            DOF ordering: [u0, v0, u1, v1, u2, v2, u3, v3]
        """
        # Material matrix (plane stress)
        D = (self.E / (1 - self.nu**2)) * np.array([
            [1,      self.nu, 0              ],
            [self.nu, 1,      0              ],
            [0,      0,      (1-self.nu)/2   ]
        ])
        
        # Gauss points and weights (2×2 quadrature)
        gauss = 1.0 / np.sqrt(3.0)
        gp = np.array([[-gauss, -gauss],
                       [ gauss, -gauss],
                       [ gauss,  gauss],
                       [-gauss,  gauss]])
        gw = np.array([1.0, 1.0, 1.0, 1.0])
        
        # Element dimensions
        a = self.elem_width
        b = self.elem_height
        
        # Initialize element stiffness
        Ke = np.zeros((8, 8))
        
        # Loop over Gauss points
        for i in range(4):
            xi, eta = gp[i]
            w = gw[i]
            
            # Shape function derivatives in natural coordinates
            dN_xi = 0.25 * np.array([
                -(1-eta),  (1-eta),  (1+eta), -(1+eta)
            ])
            dN_eta = 0.25 * np.array([
                -(1-xi), -(1+xi),  (1+xi),  (1-xi)
            ])
            
            # Jacobian matrix
            J = np.array([
                [a/2, 0  ],
                [0,   b/2]
            ])
            
            detJ = a * b / 4.0
            J_inv = np.linalg.inv(J)
            
            # Shape function derivatives in physical coordinates
            dN = J_inv @ np.array([dN_xi, dN_eta])
            dN_dx = dN[0, :]
            dN_dy = dN[1, :]
            
            # Strain-displacement matrix B (3×8)
            B = np.zeros((3, 8))
            for node in range(4):
                B[0, 2*node]     = dN_dx[node]
                B[1, 2*node + 1] = dN_dy[node]
                B[2, 2*node]     = dN_dy[node]
                B[2, 2*node + 1] = dN_dx[node]
            
            # Add contribution to element stiffness
            Ke += w * self.t * detJ * (B.T @ D @ B)
        
        return Ke
    
    def set_xi(self, xi):
        """Set element densities (design variables)"""
        self.xi = np.clip(xi, self.xi_min, 1.0)
    
    def apply_boundary_condition(self, node_ids, dof_x=True, dof_y=True):
        """
        Fix nodes (set displacements to zero)
        
        Parameters:
        -----------
        node_ids : array-like
            Node IDs to fix
        dof_x : bool
            Fix x-direction displacement
        dof_y : bool
            Fix y-direction displacement
        """
        for node in node_ids:
            if dof_x:
                self.fixed_dofs.append(2*node)
            if dof_y:
                self.fixed_dofs.append(2*node + 1)
        
        self.fixed_dofs = list(set(self.fixed_dofs))  # Remove duplicates
    
    def apply_force(self, node_id, fx=0.0, fy=0.0):
        """Apply force at a node"""
        self.forces[2*node_id]     += fx
        self.forces[2*node_id + 1] += fy
    
    def assemble_global_stiffness(self):
        """
        Assemble global stiffness matrix using element densities
        
        K_global = Σ_e (ρ_e^p) * Ke
        
        Returns sparse CSR matrix
        """
        # Prepare sparse matrix data
        row_indices = []
        col_indices = []
        values = []
        
        for elem in range(self.n_elements):
            # Get element nodes
            nodes = self.connectivity[elem]
            
            # Element DOFs (8 DOFs: 2 per node)
            elem_dofs = np.zeros(8, dtype=int)
            for i, node in enumerate(nodes):
                elem_dofs[2*i]   = 2*node      # x-displacement
                elem_dofs[2*i+1] = 2*node + 1  # y-displacement
            
            # SIMP interpolation: E_e = E_min + (E_0 - E_min) * xi^p
            density_factor = self.xi[elem] ** self.penal
            
            # Scaled element stiffness
            Ke_scaled = density_factor * self.Ke
            
            # Add to global system
            for i in range(8):
                for j in range(8):
                    row_indices.append(elem_dofs[i])
                    col_indices.append(elem_dofs[j])
                    values.append(Ke_scaled[i, j])
        
        # Create sparse matrix
        K = coo_matrix((values, (row_indices, col_indices)), 
                       shape=(self.n_dofs, self.n_dofs))
        
        return K.tocsr()
    
    def solve(self):
        """
        Solve FEA system: K*U = F
        
        Returns:
        --------
        U : array
            Global displacement vector
        is_valid : bool
            Whether solution succeeded
        """
        # Assemble global stiffness
     
        K = self.assemble_global_stiffness()
        
        # Free DOFs
        free_dofs = np.setdiff1d(np.arange(self.n_dofs), self.fixed_dofs)
        
        # Extract free system
        K_free = K[free_dofs, :][:, free_dofs]
        F_free = self.forces[free_dofs]
        
        try:
            # Solve
            d_free = spsolve(K_free, F_free)
            
            # Assemble full displacement vector
            self.displacements = np.zeros(self.n_dofs)
            self.displacements[free_dofs] = d_free
            
            # Compute compliance: c = F^T * U = U^T * K * U
            self.compliance = (self.forces @ self.displacements)
            return self.displacements, True
        except:
            return None, False
    
    def get_element_displacements(self, elem_id):
        """Get displacement vector for element elem_id"""
        if self.displacements is None:
            raise RuntimeError("Must solve first!")
        
        nodes = self.connectivity[elem_id]
        U_elem = np.zeros(8)
        
        for i, node in enumerate(nodes):
            U_elem[2*i]   = self.displacements[2*node]
            U_elem[2*i+1] = self.displacements[2*node + 1]
        
        return U_elem
    
    def get_element_strain_energy(self, elem_id):
        """
        Compute strain energy for element elem_id
        
        Returns:
        --------
        strain_energy : float
            U_e^T * Ke * U_e
        """
        d_elem = self.get_element_displacements(elem_id)
        
        # Strain energy (without density scaling for QAOA Hamiltonian)
        strain_energy = d_elem @ self.Ke @ d_elem
        
        return strain_energy
    
    def compute_elem_strain_energies(self):
        se = np.zeros(self.n_elements)
        ke = self.Ke
        u = self.displacements
        elem_dofs = np.zeros(8, dtype=int)
        for e in range(self.n_elements):
            # Map element displacements using the edof matrix
            nodes = self.connectivity[e]
            
            # Element DOFs (8 DOFs: 2 per node)
            for i, node in enumerate(nodes):
                elem_dofs[2*i]   = 2*node      # x-displacement
                elem_dofs[2*i+1] = 2*node + 1  # y-displacement
            u_e = u[elem_dofs]
            se[e] = u_e.T @ ke @ u_e
            
        return se
    def get_compliance(self):
        """Get current compliance"""
        if self.compliance is None:
            raise RuntimeError("Must solve first!")
        
        return self.compliance
    
    def evaluate_design(self):
        """Evaluate current design"""
        volume = np.sum(self.xi) * self.elem_width * self.elem_height * self.t
        
        return {
            'compliance': self.compliance,
            'volume': volume,
            'volume_fraction': np.mean(self.xi),
            'feasible': True
        }
    
    def print_metrics(self, metrics):
        """
        Print performance metrics for the current design.
        """
        print(f"  Compliance: {metrics['compliance']:.4g}")
        print(f"  Volume: {metrics['volume']:.4g}")
        print(f"  Volume fraction: {metrics['volume_fraction']:.4f}")
        print(f"  Feasible: {metrics['feasible']}")

    def plot_mesh(self, show_bc=True, show_loads=True, ax=None):
        """Plot the mesh, boundary conditions, and loads (undeformed)"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        # Plot elements

        min_xi_less_than_one = np.any(np.min(self.xi) < 1)
        for elem in range(self.n_elements):
            nodes = self.connectivity[elem]
            x = self.node_coords[nodes, 0]
            y = self.node_coords[nodes, 1]
            # Close the loop
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            ax.plot(x, y, 'k-', linewidth=0.5)
            # Fill element with color based on density xi
            if (min_xi_less_than_one):
                color = plt.cm.gray_r(self.xi[elem])
                ax.fill(x, y, color=color, alpha=0.7, edgecolor=None)

        # Plot boundary conditions (fixed DOFs)
        if show_bc and self.fixed_dofs:
            fixed_nodes = set([dof // 2 for dof in self.fixed_dofs])
            ax.plot(self.node_coords[list(fixed_nodes), 0],
                    self.node_coords[list(fixed_nodes), 1],
                    'ks', markersize=10, markerfacecolor='black', label='Fixed')

        # Plot loads
        if show_loads and np.any(np.abs(self.forces) > 1e-12):
            arrow_scale = 0.1 * max(self.lx, self.ly)
            max_force = np.max(np.abs(self.forces)) if np.any(self.forces != 0) else 1.0
            for node in range(self.n_nodes):
                fx = self.forces[2*node]
                fy = self.forces[2*node+1]
                if abs(fx) > 1e-12 or abs(fy) > 1e-12:
                    mag = np.sqrt(fx**2 + fy**2)
                    if mag == 0:
                        continue
                    dx = arrow_scale * fx / max_force
                    dy = arrow_scale * fy / max_force
                    # The arrow starts at the node and points in the direction of the force
                    ax.arrow(self.node_coords[node, 0],
                        self.node_coords[node, 1],
                        dx, dy,
                        head_width=0.03*max(self.lx, self.ly),
                        head_length=0.05*max(self.lx, self.ly),
                        fc='red', ec='red', linewidth=2, zorder=5)
                    ax.text(self.node_coords[node, 0] + dx,
                        self.node_coords[node, 1] + dy,
                        f'{mag:.2g}', color='red', fontsize=9, ha='center', va='center')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(self.title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return 
    

    def plot_displacement(self,  ax=None):
        """Plot deformed mesh"""
        if self.displacements is None:
            raise RuntimeError("Must solve first!")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        
        scale = 0.1*max(self.lx, self.ly)/np.max(np.abs(self.displacements))
        # Plot deformed mesh (colored by density)
        for elem in range(self.n_elements):
            nodes = self.connectivity[elem]
            
            # Deformed coordinates
            x = self.node_coords[nodes, 0] + scale * self.displacements[2*nodes]
            y = self.node_coords[nodes, 1] + scale * self.displacements[2*nodes + 1]
            
            # Close the loop
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            
            color = plt.cm.gray_r(self.xi[elem])
            ax.plot(x, y, color=color, linewidth=0.5)
        
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Deformed Shape (scale={scale:0.2g})')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return

class PlaneStressOC:
    """
    Optimality Criteria (OC) optimizer compatible with PlaneStressFEM.
    """
    
    def __init__(self, fea, volume_fraction, filter_radius=1.5, 
                 move_limit=0.2, max_iter=100, tol=0.01):
        self.fea = fea
        self.vf = volume_fraction
        self.rmin = filter_radius
        self.move = move_limit
        self.max_iter = max_iter
        self.tol = tol
        
        # Initialize with uniform density at target volume fraction
        self.fea.set_xi(np.ones(fea.n_elements) * volume_fraction)
        
        # Prepare filter weights based on unit element coordinates
        self._prepare_filter()
        
        self.history = {'iteration': [], 'compliance': [], 'volume_fraction': [], 'change': []}
    
    def _prepare_filter(self):
        """Prepare sensitivity filter weights assuming unit-sized elements."""
        nelx, nely = self.fea.nelx, self.fea.nely
        self.H = np.zeros((self.fea.n_elements, self.fea.n_elements))
        
        for e1 in range(self.fea.n_elements):
            ely1, elx1 = divmod(e1, nelx)
            cx1, cy1 = elx1 + 0.5, ely1 + 0.5
            
            # Find neighbors within radius rmin
            i_min, i_max = int(max(elx1 - self.rmin, 0)), int(min(elx1 + self.rmin + 1, nelx))
            j_min, j_max = int(max(ely1 - self.rmin, 0)), int(min(ely1 + self.rmin + 1, nely))
            
            for ely2 in range(j_min, j_max):
                for elx2 in range(i_min, i_max):
                    e2 = ely2 * nelx + elx2
                    dist = np.sqrt((elx1 - elx2)**2 + (ely1 - ely2)**2)
                    if dist < self.rmin:
                        self.H[e1, e2] = self.rmin - dist
        
        self.Hs = np.sum(self.H, axis=1)

    def _compute_sensitivities(self):
        """Compute compliance sensitivities dC/dxi."""
        dc = np.zeros(self.fea.n_elements)
        p = 3.0  # SIMP penalty
        ke = self.fea.Ke
        u = self.fea.displacements
        elem_dofs = np.zeros(8, dtype=int)
        for e in range(self.fea.n_elements):
            # Map element displacements using the edof matrix
            nodes = self.fea.connectivity[e]
            
            # Element DOFs (8 DOFs: 2 per node)
            for i, node in enumerate(nodes):
                elem_dofs[2*i]   = 2*node      # x-displacement
                elem_dofs[2*i+1] = 2*node + 1  # y-displacement
            u_e = u[elem_dofs]
            strain_energy = u_e.T @ ke @ u_e
            
            # Sensitivity formula: -p * xi^(p-1) * u^T * ke * u
            dc[e] = -p * (self.fea.xi[e]**(p - 1)) * strain_energy
            
        return dc

    def _filter_sensitivities(self, dc):
        """Apply density-based sensitivity filter."""
        # Standard sensitivity filter: (H * (xi * dc)) / (xi * Hs)
        dc_filtered = (self.H @ (self.fea.xi * dc)) / (self.fea.xi * self.Hs)
        return dc_filtered

    def _oc_update(self, dc):
        """Optimality Criteria update step with bisection for Lagrange multiplier."""
        l1, l2 = 0, 1e9
        xi_old = self.fea.xi.copy()
        
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            # OC update rule with move limits and physical bounds [0, 1]
            xi_new = np.maximum(0.001, # Avoid zero for numerical stability
                     np.maximum(xi_old - self.move,
                     np.minimum(1.0,
                     np.minimum(xi_old + self.move,
                                xi_old * np.sqrt(-dc / lmid)))))
            
            # Check volume constraint (average density)
            if np.mean(xi_new) > self.vf:
                l1 = lmid
            else:
                l2 = lmid
        return xi_new

    def optimize(self, verbose=True):
        if verbose:
            print(f"Starting OC Optimization: Target VF={self.vf}, Filter R={self.rmin}")

        for it in range(self.max_iter):
            # 1. FEA Solve
            self.fea.solve()
            
            # 2. Get Metrics and Sensitivities
            metrics = self.fea.evaluate_design()
            dc = self._compute_sensitivities()
            
            # 3. Filter and Update
            dc_filtered = self._filter_sensitivities(dc)
            xi_old = self.fea.xi.copy()
            xi_new = self._oc_update(dc_filtered)
            
            # 4. Update FEM state
            self.fea.set_xi(xi_new)
            
            # 5. Convergence check
            change = np.max(np.abs(xi_new - xi_old))
            self.history['iteration'].append(it)
            self.history['compliance'].append(metrics['compliance'])
            self.history['volume_fraction'].append(metrics['volume_fraction'])
            self.history['change'].append(change)

            if verbose and it % 5 == 0:
                print(f"Iter {it:3d}: Compliance={metrics['compliance']:.4g}, volume_fraction={metrics['volume_fraction']:.4f}, Change={change:.4f}")

            
        
        return self.history

"""
Examples of truss problems

"""

def truss2x2(E=200e9, A=0.0005):
    # Example usage with the 2x2 truss
    nodes = np.array([
        [0.0, 0.0],   # Node 0 (bottom left) - FIXED
        [1.0, 0.0],   # Node 1 (bottom right)  - LOADED
        [0.0, 1.0],   # Node 2 (top left) - FIXED
        [1.0, 1.0]    # Node 3 (top right)
    ])

    elements = [
        (0, 1), (2, 3),  # Horizontal
        (0, 2), (1, 3),  # Vertical
        (0, 3), (1, 2)   # Diagonals
    ]

    # Define problem
    fixed_dofs = [0, 1, 4, 5]  # Nodes 0 and 2 fixed
    loads = np.zeros(2 * len(nodes))
    loads[2*1 + 1] = -100000  # 100 kN downward at node 1

     # Create FEM model
    fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=E, A=A)

    return fem_model

def truss2x3(E=200e9, A=0.0005):
    # Example usage with the 2x3 truss
    nodes = np.array([
        [0.0, 0.0],   # Node 0 (bottom left) - FIXED
        [1.0, 0.0],   # Node 1 (bottom right) - LOADED
        [0.0, 1.0],   # Node 2 (mid left)  - FIXED
        [1.0, 1.0],   # Node 3 (mid right) 
        [0.0, 2.0],   # Node 4 (top left)  - FIXED
        [1.0, 2.0]    # Node 5 (top right)
    ])

    elements = [
        (0, 1), (0, 2), (0, 3), 
        (1, 2), (1, 3),
        (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5),
        (4, 5)
    ]

    # Define problem
    fixed_dofs = [0, 1, 4, 5, 8, 9]  # Nodes 0, 2, and 4 fixed
    loads = np.zeros(2 * len(nodes))
    loads[2*1 + 1] = -10000  # 10 kN downward at node 1

    # Create FEM model
    fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=E, A=A)

    return fem_model


def truss3x2(E=200e9, A=0.0005):
    # Example usage with the 3x2 truss
    # Nodes: 6
    # Elements: 11
    nodes = np.array([
        [0.0, 0.0],   # Node 0 (bottom left) - FIXED
        [2.0, 0.0],   # Node 1 (bottom center)
        [4.0, 0.0],   # Node 2 (bottom right) - FIXED
        [0.0, 2.0],   # Node 3 (top left)
        [2.0, 2.0],   # Node 4 (top center) - LOADED
        [4.0, 2.0]    # Node 5 (top right)
    ])

    elements = [
        (0, 1), (1, 2), (3, 4), (4, 5),  # Horizontal
        (0, 3), (1, 4), (2, 5),          # Vertical
        (0, 4), (1, 3), (1, 5), (2, 4)   # Diagonals
    ]

    # Define problem
    fixed_dofs = [0, 1, 4, 5]  # Nodes 0 and 2 fixed
    loads = np.zeros(2 * len(nodes))
    loads[2*4 + 1] = -10000  # 10 kN downward at node 4

    # Create FEM model
    fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=E, A=A)

    return fem_model

def truss3x3(E=200e9, A=0.0005):
    # Example usage with the 3x3 truss
    # Nodes: 9
    # Elements: 26
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
    fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=E, A=A)

    return fem_model

def truss_grid(M = 8, N = 4, Lx=1.0, E=200e9, A=0.0005):
        """
        Create a 2D truss ground structure with an M x N grid of nodes.
        Each square cell is fully connected: horizontal, vertical, and both diagonals.

        Parameters:
        -----------
        M : int
            Number of nodes in x-direction (columns)
        N : int
            Number of nodes in y-direction (rows)
        Lx : float
            Total width of the grid
        Ly : float
            Total height of the grid
        E : float
            Young's modulus
        A : float
            Cross-sectional area

        Returns:
        --------
        TrussFEM instance
        """
        # Node coordinates
        Ly= Lx * (N - 1) / (M - 1)  # Maintain aspect ratio
        x_coords = np.linspace(0, Lx, M)
        y_coords = np.linspace(0, Ly, N)
        nodes = np.array([[x, y] for y in y_coords for x in x_coords])

        # Helper to get node index from (ix, iy)
        def node_id(ix, iy):
            return iy * M + ix

        elements = []
        for iy in range(N):
            for ix in range(M):
                n0 = node_id(ix, iy)
                # Horizontal (right neighbor)
                if ix < M - 1:
                    n1 = node_id(ix + 1, iy)
                    elements.append((n0, n1))
                # Vertical (top neighbor)
                if iy < N - 1:
                    n2 = node_id(ix, iy + 1)
                    elements.append((n0, n2))
                # Diagonal up-right
                if ix < M - 1 and iy < N - 1:
                    n3 = node_id(ix + 1, iy + 1)
                    elements.append((n0, n3))
                # Diagonal up-left
                if ix > 0 and iy < N - 1:
                    n4 = node_id(ix - 1, iy + 1)
                    elements.append((n0, n4))

        # Example boundary conditions: fix left wall,
        fixed_dofs = []
        for iy in range(N):
            n = node_id(0, iy)
            fixed_dofs.extend([2 * n, 2 * n + 1])
        print(f"Fixed DOFs at left wall nodes: {[node_id(0, iy) for iy in range(N)]}")
        loads = np.zeros(2 * M * N)
        bottom_right = node_id(M - 1, 0)
        print(f"Applying load at node {bottom_right} (bottom right corner)")
        loads[2 * bottom_right + 1] = -10000  # Downward force at bottom right node

        fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=E, A=A)
        return fem_model

def truss3x3Substructure(E=200e9, A=0.0005):
    
    fem_model = truss3x3(E=E, A=A)
    #  Plot with a specific design
    sub_structure = np.array([
        # Horizontal members (indices 0-5)
        1, 1,  # Bottom row: (0,1), (1,2) - CRITICAL for node 1 stability
        1, 1,  # Middle row: (3,4), (4,5) - lateral bracing
        0, 0,  # Top row: not needed (nodes 6,8 hanging)
        
        # Vertical members (indices 6-11)
        1, 0,  # Left column: (0,3) active
        1, 1,  # Center column: (1,4), (4,7) - load path
        1, 0,  # Right column: (2,5) active
        
        # In-square diagonals (indices 12-19)
        1, 0,  # (0,4) diagonal bracing
        0, 1,  # (2,4) diagonal bracing
        1, 0,  # (3,7) diagonal bracing
        0, 1,  # (5,7) diagonal bracing
        
        # Long diagonals (indices 20-25)
        0, 0, 0, 0, 0, 0
    ], dtype=int)
    area = fem_model.A * sub_structure  # Zero area for inactive members
    fem_model.set_area(area)
    print(f"Number of active members: {np.sum(sub_structure)}")  # 10 members
    return fem_model

def truss_10bar(E=200e9, A=0.005):
    """
    Classic 10-bar truss ground structure
    
    Reference: Rajeev, S., & Krishnamoorthy, C. S. (1992). 
    "Discrete optimization of structures using genetic algorithms."
    Journal of Structural Engineering, 118(5), 1233-1250.
    
    Node layout:
    - Bottom row (y=0): Nodes 0, 1, 2
    - Top row (y=H):   Nodes 3, 4, 5
    
    Boundary conditions:
    - Node 0: Fixed (bottom left)
    - Node 2: Fixed (bottom right)
    
    Loading:
    - Node 1: Vertical load (bottom center)
    - Node 3: Vertical load (top left)
    
    Ground structure: 10 potential members
    - 4 horizontals (top and bottom chords)
    - 2 verticals (left and right, NO center vertical)
    - 4 diagonals (cross-bracing)
    """
    
    # Standard dimensions from benchmark (360 inches = 9.14 m)
    L = 9.14  # Bay width (meters)
    H = 9.14  # Height (meters)
    
    # Node coordinates
    nodes = np.array([
        [0.0, 0.0],   # Node 0 (bottom left) - FIXED
        [L,   0.0],   # Node 1 (bottom center) - LOADED
        [2*L, 0.0],   # Node 2 (bottom right) - FIXED
        [0.0, H],     # Node 3 (top left) - LOADED
        [L,   H],     # Node 4 (top center)
        [2*L, H]      # Node 5 (top right)
    ])
    
    # Ground structure with exactly 10 potential bars
    elements = [
        (0, 1),  # Bar 0: Bottom chord left
        (1, 2),  # Bar 1: Bottom chord right
        (3, 4),  # Bar 2: Top chord left
        (4, 5),  # Bar 3: Top chord right
        (0, 3),  # Bar 4: Vertical left
        (2, 5),  # Bar 5: Vertical right
        (0, 4),  # Bar 6: Diagonal (0→4)
        (1, 3),  # Bar 7: Diagonal (1→3)
        (1, 5),  # Bar 8: Diagonal (1→5)
        (2, 4)   # Bar 9: Diagonal (2→4)
    ]
    
    # Fixed degrees of freedom
    fixed_dofs = [0, 1, 4, 5]  # Nodes 0 and 2 fully fixed (x and y)
    
    # Applied loads (standard benchmark loads)
    loads = np.zeros(2 * len(nodes))
    P = 444822  # 100 kips = 444,822 N
    loads[2*1 + 1] = -P  # Downward load at node 1 (bottom center)
    loads[2*3 + 1] = -P  # Downward load at node 3 (top left)
    
    # Create FEM model
    fem_model = TrussFEM(nodes, elements, loads, fixed_dofs, E=E, A=A)
    
    return fem_model

"""
Examples of plane stress problems

"""

def PlaneStressCantilever(nx=60, ny=20, E= 200e9, nu=0.3):
    """
    Classic cantilever beam topology optimization problem
    
    Setup:
        |============================
        |                           ↓ F
        |============================
        
    Fixed left edge, load at center of right edge
    """
   
    # Create mesh
    ly = 1.0/ny
    lx = nx * (ly / ny)

    fea2D = PlaneStressFEM(nx=nx, ny=ny, lx=lx, ly=ly, E=E, nu=nu)
    
    # Boundary conditions: Fix left edge
    left_nodes = np.arange(0, fea2D.n_nodes, fea2D.n_nodes_x)
    fea2D.apply_boundary_condition(left_nodes, dof_x=True, dof_y=True)
    
    # Load: Downward force at center of right edge
    center_right_node = fea2D.n_nodes_x - 1 + (ny // 2) * fea2D.n_nodes_x
    fea2D.apply_force(center_right_node, fx=0.0, fy=-10000)
    fea2D.title = "Plane Stress Cantilever Beam"
    return fea2D


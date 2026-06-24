#!/usr/bin/env python3
"""
================================================================================
SÜNDERHAUF ORACLE-BASED BLOCK-ENCODING
================================================================================

Implementation of the oracle-based block-encoding method from:

    Sünderhauf, Campbell, and Camps (2024)
    "Block-encoding structured matrices for data input in quantum computing"
    Quantum, vol. 8, p. 1226

Key Idea:
    Instead of storing all matrix entries, store only D distinct values
    and use oracles to compute which entry goes where.

Labeling:
    Each non-zero A[i,j] is labeled by (d, m):
        d = dictionary index (which value, 0 to D-1)
        m = multiplicity index (which occurrence of that value)

Oracles:
    Oc: (d,m) → (j, sc)   Column oracle
    Or: (d,m) → (i, sr)   Row oracle  
    Org: (d,m) → {0,1}    Out-of-range flag

Resources:
    Ancilla qubits: 2 + ceil(log2(S))
    Data loading: D values
    
================================================================================
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass


# ============================================================================
# CORE SÜNDERHAUF IMPLEMENTATION
# ============================================================================

@dataclass
class SunderhaufEncoding:
    """Container for Sünderhauf block-encoding parameters."""
    D: int                          # Number of distinct values
    S: int                          # Maximum sparsity
    M: int                          # Maximum multiplicity
    values: List[float]             # The D distinct values
    alpha_base: float               # Base subnormalization
    alpha_prep: float               # PREP/UNPREP subnormalization
    ancilla_qubits: int             # Required ancillas


def analyze_matrix(A: np.ndarray, tol: float = 1e-10) -> SunderhaufEncoding:
    """
    Analyze a matrix for Sünderhauf block-encoding.
    
    Parameters:
        A: Input matrix (dense numpy array)
        tol: Tolerance for identifying distinct values
        
    Returns:
        SunderhaufEncoding with all parameters
    """
    N = A.shape[0]
    
    # Find distinct non-zero values
    nonzeros = A[np.abs(A) > tol]
    values = []
    counts = {}
    
    for val in nonzeros:
        found = False
        for v in values:
            if abs(val - v) < tol:
                counts[v] += 1
                found = True
                break
        if not found:
            values.append(val)
            counts[val] = 1
    
    D = len(values)
    M = max(counts.values()) if counts else 0
    
    # Compute sparsity
    Sr = max(np.sum(np.abs(A) > tol, axis=1))  # Max non-zeros per row
    Sc = max(np.sum(np.abs(A) > tol, axis=0))  # Max non-zeros per column
    S = max(Sr, Sc)
    
    # Subnormalization
    A_max = max(abs(v) for v in values)
    alpha_base = np.sqrt(Sr * Sc) * A_max
    
    if D <= S:
        alpha_prep = (np.sqrt(Sr * Sc) / D) * sum(abs(v) for v in values)
    else:
        alpha_prep = float('inf')
    
    # Ancilla qubits
    ancilla = 2 + int(np.ceil(np.log2(S)))
    
    return SunderhaufEncoding(
        D=D, S=S, M=M,
        values=values,
        alpha_base=alpha_base,
        alpha_prep=alpha_prep,
        ancilla_qubits=ancilla
    )


# ============================================================================
# EXAMPLE 1: 1D LAPLACIAN (TRIDIAGONAL)
# ============================================================================

class Tridiagonal1D:
    """
    1D Laplacian: A = tridiag(-1, 2, -1)
    
    Matrix structure:
        [ 2  -1   0   0   0 ]
        [-1   2  -1   0   0 ]
        [ 0  -1   2  -1   0 ]
        [ 0   0  -1   2  -1 ]
        [ 0   0   0  -1   2 ]
    
    Dictionary: D = 2
        d=0: diagonal value (2)
        d=1: off-diagonal value (-1)
    
    Sparsity: S = 3 (at most 3 non-zeros per row)
    """
    
    def __init__(self, N: int):
        self.N = N
        self.D = 2
        self.S = 3
        self.values = [2.0, -1.0]  # d=0: diagonal, d=1: off-diagonal
        
    def oracle_Oc(self, d: int, m: int) -> Tuple[int, int]:
        """
        Column oracle: (d,m) → (j, sc)
        
        For d=0 (diagonal): m is the row/column index
        For d=1 (off-diagonal): m encodes position and direction
        """
        if d == 0:  # Diagonal
            j = m
            sc = 0
        elif d == 1:  # Off-diagonal
            row = m // 2
            direction = m % 2  # 0 = left, 1 = right
            j = row - 1 if direction == 0 else row + 1
            sc = 1 if direction == 0 else 2
        else:
            raise ValueError(f"Invalid d={d}")
        return j, sc
    
    def oracle_Or(self, d: int, m: int) -> Tuple[int, int]:
        """Row oracle: (d,m) → (i, sr)"""
        if d == 0:
            i = m
            sr = 0
        elif d == 1:
            i = m // 2
            sr = 1 if m % 2 == 0 else 2
        else:
            raise ValueError(f"Invalid d={d}")
        return i, sr
    
    def oracle_Org(self, d: int, m: int) -> bool:
        """Out-of-range oracle: True if (d,m) is invalid"""
        if d == 0:
            return m < 0 or m >= self.N
        elif d == 1:
            row = m // 2
            direction = m % 2
            if row < 0 or row >= self.N:
                return True
            if direction == 0 and row == 0:  # No left neighbor for first row
                return True
            if direction == 1 and row == self.N - 1:  # No right for last row
                return True
            return False
        return True
    
    def get_value(self, d: int) -> float:
        """Get the value for dictionary index d"""
        return self.values[d]
    
    def assemble(self) -> np.ndarray:
        """Assemble the full matrix"""
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            A[i, i] = 2.0
            if i > 0:
                A[i, i-1] = -1.0
            if i < self.N - 1:
                A[i, i+1] = -1.0
        return A
    
    def verify(self) -> bool:
        """Verify oracles produce correct matrix"""
        A = self.assemble()
        A_reconstructed = np.zeros_like(A)
        
        for d in range(self.D):
            for m in range(2 * self.N):  # Upper bound on m
                if self.oracle_Org(d, m):
                    continue
                i, _ = self.oracle_Or(d, m)
                j, _ = self.oracle_Oc(d, m)
                if 0 <= i < self.N and 0 <= j < self.N:
                    A_reconstructed[i, j] = self.get_value(d)
        
        return np.allclose(A, A_reconstructed)


# ============================================================================
# EXAMPLE 2: 2D LAPLACIAN (FIVE-POINT STENCIL)
# ============================================================================

class Laplacian2D:
    """
    2D Laplacian with five-point stencil on Nx × Ny grid.
    
    Stencil:
              N
              |
        W --- C --- E
              |
              S
    
    Matrix entry pattern:
        A[i,i] = 4           (center)
        A[i,i±1] = -1        (east/west, if not at boundary)
        A[i,i±Nx] = -1       (north/south, if not at boundary)
    
    Dictionary: D = 2 (or D = 3 if dx ≠ dy)
        d=0: diagonal (4)
        d=1: off-diagonal (-1)
    
    Sparsity: S = 5
    """
    
    def __init__(self, Nx: int, Ny: int):
        self.Nx = Nx
        self.Ny = Ny
        self.N = Nx * Ny
        self.D = 2
        self.S = 5
        self.values = [4.0, -1.0]
        
    def _node_to_coords(self, i: int) -> Tuple[int, int]:
        """Convert linear index to (x, y) coordinates"""
        return i % self.Nx, i // self.Nx
    
    def _coords_to_node(self, x: int, y: int) -> int:
        """Convert (x, y) coordinates to linear index"""
        return y * self.Nx + x
    
    def oracle_Oc(self, d: int, m: int) -> Tuple[int, int]:
        """
        Column oracle: (d,m) → (j, sc)
        
        For d=0 (diagonal): m is the node index
        For d=1 (neighbors): m encodes node and direction
            direction: 0=W, 1=E, 2=S, 3=N
        """
        if d == 0:
            j = m
            sc = 0
        elif d == 1:
            node = m // 4
            direction = m % 4
            x, y = self._node_to_coords(node)
            
            if direction == 0:    # West
                j = self._coords_to_node(x - 1, y)
                sc = 1
            elif direction == 1:  # East
                j = self._coords_to_node(x + 1, y)
                sc = 2
            elif direction == 2:  # South
                j = self._coords_to_node(x, y - 1)
                sc = 3
            else:                 # North
                j = self._coords_to_node(x, y + 1)
                sc = 4
        else:
            raise ValueError(f"Invalid d={d}")
        return j, sc
    
    def oracle_Or(self, d: int, m: int) -> Tuple[int, int]:
        """Row oracle: (d,m) → (i, sr)"""
        if d == 0:
            return m, 0
        elif d == 1:
            node = m // 4
            direction = m % 4
            return node, direction + 1
        raise ValueError(f"Invalid d={d}")
    
    def oracle_Org(self, d: int, m: int) -> bool:
        """Out-of-range oracle"""
        if d == 0:
            return m < 0 or m >= self.N
        elif d == 1:
            node = m // 4
            if node < 0 or node >= self.N:
                return True
            
            direction = m % 4
            x, y = self._node_to_coords(node)
            
            if direction == 0 and x == 0:           # West boundary
                return True
            if direction == 1 and x == self.Nx - 1: # East boundary
                return True
            if direction == 2 and y == 0:           # South boundary
                return True
            if direction == 3 and y == self.Ny - 1: # North boundary
                return True
            return False
        return True
    
    def get_value(self, d: int) -> float:
        return self.values[d]
    
    def assemble(self) -> np.ndarray:
        """Assemble full matrix"""
        A = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            x, y = self._node_to_coords(i)
            A[i, i] = 4.0
            
            if x > 0:
                A[i, i - 1] = -1.0
            if x < self.Nx - 1:
                A[i, i + 1] = -1.0
            if y > 0:
                A[i, i - self.Nx] = -1.0
            if y < self.Ny - 1:
                A[i, i + self.Nx] = -1.0
        
        return A
    
    def verify(self) -> bool:
        """Verify oracles"""
        A = self.assemble()
        A_rec = np.zeros_like(A)
        
        for d in range(self.D):
            for m in range(5 * self.N):
                if self.oracle_Org(d, m):
                    continue
                i, _ = self.oracle_Or(d, m)
                j, _ = self.oracle_Oc(d, m)
                if 0 <= i < self.N and 0 <= j < self.N:
                    A_rec[i, j] = self.get_value(d)
        
        return np.allclose(A, A_rec)


# ============================================================================
# SUBNORMALIZATION
# ============================================================================

def compute_subnormalization(values: List[float], Sr: int, Sc: int, D: int, S: int):
    """
    Compute subnormalization factors for Sünderhauf encoding.
    
    Base scheme (always applicable):
        α = √(Sc × Sr) × max|A_d|
    
    PREP/UNPREP scheme (requires D ≤ S):
        α = (√(Sc × Sr) / D) × Σ|A_d|
    """
    A_max = max(abs(v) for v in values)
    sum_abs = sum(abs(v) for v in values)
    
    alpha_base = np.sqrt(Sc * Sr) * A_max
    
    if D <= S:
        alpha_prep = (np.sqrt(Sc * Sr) / D) * sum_abs
        improvement = (1 - alpha_prep / alpha_base) * 100
    else:
        alpha_prep = None
        improvement = None
    
    return alpha_base, alpha_prep, improvement


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo_1d_laplacian():
    """Demonstrate Sünderhauf encoding for 1D Laplacian."""
    
    print("=" * 60)
    print("EXAMPLE 1: 1D LAPLACIAN (TRIDIAGONAL)")
    print("=" * 60)
    
    N = 5
    lap1d = Tridiagonal1D(N)
    A = lap1d.assemble()
    
    print("\nMatrix A (5×5 tridiagonal):")
    print(A)
    
    print("\n--- Sünderhauf Parameters ---")
    print(f"Dictionary size D = {lap1d.D}")
    print(f"  d=0: value = {lap1d.values[0]} (diagonal)")
    print(f"  d=1: value = {lap1d.values[1]} (off-diagonal)")
    print(f"Sparsity S = {lap1d.S}")
    print(f"Ancilla qubits = 2 + ⌈log₂({lap1d.S})⌉ = {2 + int(np.ceil(np.log2(lap1d.S)))}")
    
    print("\n--- Oracle Examples ---")
    print("Format: (d, m) → i, j, value")
    
    examples = [
        (0, 0), (0, 2), (0, 4),  # Diagonal entries
        (1, 1), (1, 2), (1, 5),  # Off-diagonal entries
    ]
    
    for d, m in examples:
        if lap1d.oracle_Org(d, m):
            print(f"  (d={d}, m={m}) → OUT OF RANGE")
        else:
            i, _ = lap1d.oracle_Or(d, m)
            j, _ = lap1d.oracle_Oc(d, m)
            val = lap1d.get_value(d)
            print(f"  (d={d}, m={m}) → A[{i},{j}] = {val}")
    
    print(f"\nOracle verification: {'PASSED ✓' if lap1d.verify() else 'FAILED ✗'}")
    
    # Subnormalization
    alpha_base, alpha_prep, improvement = compute_subnormalization(
        lap1d.values, Sr=3, Sc=3, D=lap1d.D, S=lap1d.S
    )
    print(f"\n--- Subnormalization ---")
    print(f"Base scheme:     α = {alpha_base:.4f}")
    print(f"PREP/UNPREP:     α = {alpha_prep:.4f}")
    print(f"Improvement:     {improvement:.1f}%")


def demo_2d_laplacian():
    """Demonstrate Sünderhauf encoding for 2D Laplacian."""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: 2D LAPLACIAN (FIVE-POINT STENCIL)")
    print("=" * 60)
    
    Nx, Ny = 3, 3
    lap2d = Laplacian2D(Nx, Ny)
    A = lap2d.assemble()
    
    print(f"\nGrid: {Nx}×{Ny} = {lap2d.N} nodes")
    print("\nNode numbering:")
    print("  6 - 7 - 8")
    print("  |   |   |")
    print("  3 - 4 - 5")
    print("  |   |   |")
    print("  0 - 1 - 2")
    
    print("\nMatrix A (9×9):")
    print(A.astype(int))
    
    print("\n--- Sünderhauf Parameters ---")
    print(f"Dictionary size D = {lap2d.D}")
    print(f"  d=0: value = {lap2d.values[0]} (diagonal)")
    print(f"  d=1: value = {lap2d.values[1]} (neighbors)")
    print(f"Sparsity S = {lap2d.S}")
    print(f"Ancilla qubits = 2 + ⌈log₂({lap2d.S})⌉ = {2 + int(np.ceil(np.log2(lap2d.S)))}")
    
    print("\n--- Oracle Examples ---")
    print("Format: (d, m) → i, j, value")
    print("For d=1: m = 4×node + direction, direction: 0=W, 1=E, 2=S, 3=N")
    
    examples = [
        (0, 4),   # Diagonal of center node
        (1, 17),  # Node 4, East neighbor (4×4 + 1 = 17)
        (1, 18),  # Node 4, South neighbor (4×4 + 2 = 18)
        (1, 0),   # Node 0, West (should be out of range)
    ]
    
    for d, m in examples:
        if lap2d.oracle_Org(d, m):
            print(f"  (d={d}, m={m:2d}) → OUT OF RANGE")
        else:
            i, _ = lap2d.oracle_Or(d, m)
            j, _ = lap2d.oracle_Oc(d, m)
            val = lap2d.get_value(d)
            print(f"  (d={d}, m={m:2d}) → A[{i},{j}] = {val}")
    
    print(f"\nOracle verification: {'PASSED ✓' if lap2d.verify() else 'FAILED ✗'}")
    
    # Subnormalization
    alpha_base, alpha_prep, improvement = compute_subnormalization(
        lap2d.values, Sr=5, Sc=5, D=lap2d.D, S=lap2d.S
    )
    print(f"\n--- Subnormalization ---")
    print(f"Base scheme:     α = {alpha_base:.4f}")
    print(f"PREP/UNPREP:     α = {alpha_prep:.4f}")
    print(f"Improvement:     {improvement:.1f}%")


def demo_general_matrix():
    """Analyze a general matrix."""
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: GENERAL SPARSE MATRIX")
    print("=" * 60)
    
    # Create a sparse matrix with repeated values
    A = np.array([
        [10,  3,  0,  0,  3],
        [ 3, 10,  3,  0,  0],
        [ 0,  3, 10,  3,  0],
        [ 0,  0,  3, 10,  3],
        [ 3,  0,  0,  3, 10]
    ], dtype=float)
    
    print("\nMatrix A:")
    print(A.astype(int))
    
    encoding = analyze_matrix(A)
    
    print("\n--- Sünderhauf Analysis ---")
    print(f"Dictionary size D = {encoding.D}")
    print(f"Distinct values: {encoding.values}")
    print(f"Sparsity S = {encoding.S}")
    print(f"Max multiplicity M = {encoding.M}")
    print(f"Ancilla qubits = {encoding.ancilla_qubits}")
    print(f"\nSubnormalization:")
    print(f"  Base:      α = {encoding.alpha_base:.4f}")
    if encoding.alpha_prep < float('inf'):
        print(f"  PREP:      α = {encoding.alpha_prep:.4f}")
        improvement = (1 - encoding.alpha_prep / encoding.alpha_base) * 100
        print(f"  Improvement: {improvement:.1f}%")
    else:
        print(f"  PREP: Not applicable (D > S)")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SÜNDERHAUF ORACLE-BASED BLOCK-ENCODING")
    print("Implementation and Examples")
    print("=" * 60)
    
    demo_1d_laplacian()
    demo_2d_laplacian()
    demo_general_matrix()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Points:
    
1. (d,m) Labeling:
   - d indexes the distinct value (0 to D-1)
   - m indexes which occurrence of that value
   
2. Oracles compute matrix structure:
   - Oc(d,m) → column index j
   - Or(d,m) → row index i
   - Org(d,m) → flags invalid entries
   
3. Resources:
   - Ancilla qubits: 2 + ceil(log₂(S))
   - Data loading: D values only
   
4. Subnormalization:
   - Base: α = √(Sc·Sr) × max|A_d|
   - PREP/UNPREP: α = (√(Sc·Sr)/D) × Σ|A_d|  (requires D ≤ S)
   - Lower α = better success probability
    """)
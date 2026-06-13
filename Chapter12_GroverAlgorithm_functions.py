"""
Landmark Quantum Algorithms
============================
Implementations of foundational quantum algorithms demonstrating quantum advantage.

Algorithms Included:
--------------------
1. Bernstein-Vazirani (BV): Finds hidden bitstring in single query (vs n queries classically)
2. Grover's Search: Finds marked element in O(sqrt(N)) queries (vs O(N) classically)

Key Quantum Concepts:
---------------------
- Quantum parallelism: Evaluate function on superposition of all inputs simultaneously
- Phase kickback: Oracle marks solutions by phase flip
- Amplitude amplification: Increase probability of measuring correct answer

Computational Complexity:
-------------------------
- BV: 1 quantum query vs n classical queries (exponential speedup)
- Grover: O(sqrt(N)) vs O(N) classical (quadratic speedup)
  Optimal iterations ≈ π/4 * sqrt(N)
  
References:
-----------
- Bernstein & Vazirani (1997): Quantum complexity theory
- Grover (1996): Fast quantum mechanical algorithm for database search
- Nielsen & Chuang (2010): Quantum Computation and Quantum Information, Ch. 6
"""
import numpy as np
from IPython.display import display

from qiskit import QuantumCircuit
from qiskit.quantum_info import  Operator
from qiskit.circuit.library import  MCXGate, PhaseOracle
from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseOracleGate

from Chapter08_QuantumGates_functions import simulate_statevector,  simulate_measurements
import math

def bitstring_to_expression(bitstring_expr: str):
    """Convert bitstring or Boolean expression of bitstrings to PhaseOracleGate expression.
    
    Examples:
    - Single bitstring '100' → '~x0 & x1 & x2'
    - Multiple bitstrings '100 & 101' → '(~x0 & x1 & x2) & (~x0 & x2 & x3)'
    
    Ensures variables appear in order x0, x1, ... so parse-order
    matches index order."""
    
    def convert_single_bitstring(bitstring: str):
        """Convert single bitstring like '100' to expression.
        Rightmost bit is x0, leftmost is x(n-1)."""
        n = len(bitstring)
        terms = []
        # Reverse bitstring so we index from left-to-right after reversal
        reversed_bits = bitstring[::-1]
        for i, bit in enumerate(reversed_bits):
            terms.append(f"x{i}" if bit == '1' else f"~x{i}")
        return " & ".join(terms)
    
    import re

    expr = bitstring_expr.strip()

    # Single pure bitstring, e.g. "101"
    if re.fullmatch(r"[01]+", expr):
        return convert_single_bitstring(expr)

    # Boolean expression case: replace only binary tokens (e.g. 101, 001)
    # while preserving operators/parentheses.
    def repl(match):
        return f"({convert_single_bitstring(match.group(0))})"

    return re.sub(r"\b[01]+\b", repl, expr)

def ensure_variable_order(expression, n):
    prefix = " & ".join(f"(x{i} | ~x{i})" for i in range(n))
    return prefix + " & " + expression


def get_feasible_expression():
    one_feasible_soln  = (
        "(x0 | x1 | x2) & (x0 | x3 | x4)"
        " & (x1 | x3 | x5 | x6) & (x2 | x5) & (x4 | x6) & (~x1 | ~x4)"
        " & (~x1 | x3) & (x0 | ~x4) & (~x4 | x2) & (x2 | x3) & (x5 | x0)"
        " & (~x1 | x6) & (x6 | x2) & (~x4 | ~x1 | x5) & (x3 | x6 | ~x1)"
        " & (~x1 | x5 | x2) & (x0 | x3 | x6) & (x5 | x3 | ~x4)"
        " & (x2 | ~x4 | x6) & (~x1 | ~x4 | x0) & (x0 | ~x5 | x3)"
        " & (~x1 | ~x3 | x5) & (~x1 | x2 | ~x3) & (~x4 | x5 | ~x3)"
        " & (~x4 | ~x6 | x3) & (x5 | ~x2 | ~x3) & (x3 | ~x0 | ~x2)"
        " & (~x4 | x5 | ~x6) & (~x1 | ~x2 | ~x3 | x5)"
        " & (x0 | ~x2 | ~x3) & (x2 | ~x3 | ~x5) & (~x4 | ~x5 | x6)"
        " & (~x4 | x1 | ~x5) & (~x1 | ~x0 | ~x6)"
    )
    return one_feasible_soln


def get_void_expression(grid):
    """
    Generate a PhaseOracleGate Boolean expression for a microstructure grid
    using coordinate encoding.

    Convention:
        0 = void  (white cell)
        1 = solid (gray cell)

    Encoding:
        A 2^m x 2^m grid has N = 2^(2m) cells. Rather than one qubit per cell
        (which would require N qubits), we encode each cell by its (row, col)
        address using two m-qubit registers -- one for the row index i, one for
        the column index j. This requires only 2m qubits total: an exponential
        reduction over the flat encoding.

        Register layout (Qiskit little-endian):
            Row register : xi_0, xi_1, ..., xi_{m-1}   (qubits 0 .. m-1)
            Col register : xj_0, xj_1, ..., xj_{m-1}   (qubits m .. 2m-1)

        Each void at (i_v, j_v) produces one conjunction over all 2m variables.
        Multiple voids produce a disjunction of such conjunctions, so every void
        address is a marked state for the Grover oracle.

    Parameters
    ----------
    grid : 2D array-like of int, shape (2^m, 2^m)

    Returns
    -------
    expression   : str             Boolean expression for PhaseOracleGate
    n_qubits     : int             2m -- number of qubits required
    void_coords  : list[(int,int)] (row, col) pairs of all voids
    """
    grid_array = np.array(grid)
    rows, cols = grid_array.shape
    m = int(np.log2(rows))          # rows = cols = 2^m

    void_coords = [
        (i, j)
        for i in range(rows)
        for j in range(cols)
        if grid_array[i, j] == 0    # 0 = void
    ]

    if not void_coords:
        raise ValueError("No voids found in grid.")

    def coord_to_clause(i, j):
        """One conjunction that fires exactly at address (i, j).
        Row bits: xi_0..xi_{m-1}; Col bits: xj_0..xj_{m-1} (little-endian)."""
        row_bits = format(i, f'0{m}b')[::-1]   # reverse for little-endian
        col_bits = format(j, f'0{m}b')[::-1]
        terms = []
        for k, bit in enumerate(row_bits):
            terms.append(f"xi{k}" if bit == '1' else f"~xi{k}")
        for k, bit in enumerate(col_bits):
            terms.append(f"xj{k}" if bit == '1' else f"~xj{k}")
        return "(" + " & ".join(terms) + ")"

    clauses    = [coord_to_clause(i, j) for i, j in void_coords]
    expression = " | ".join(clauses)
    n_qubits   = 2 * m
    return expression, n_qubits, void_coords


def decode_void_measurement(bitstring, m):
    """
    Decode a Grover measurement bitstring back to (row, col) coordinates.

    The bitstring is in Qiskit's little-endian order:
        bits 0..m-1  -> row index i
        bits m..2m-1 -> col index j

    Parameters
    ----------
    bitstring : str   e.g. '0110' for m=2
    m         : int   log2 of grid side length

    Returns
    -------
    (row, col) : (int, int)
    """
    col = int(bitstring[:m], 2)   # high qubits → col
    row = int(bitstring[m:], 2)   # low  qubits → row
    return row, col

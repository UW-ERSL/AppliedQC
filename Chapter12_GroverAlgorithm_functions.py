"""
Grover's algorithm helpers for Chapter 12, "Grover Algorithm".

Utilities for building the Boolean expressions that Qiskit's
``PhaseOracleGate`` consumes, and for the two engineering searches in the
chapter:

* ``bitstring_to_expression`` -- a bitstring (or a Boolean combination of
  bitstrings) to a conjunction over x0, x1, ... in little-endian order.
* ``get_qiskit_expression``   -- pad an expression with tautologies so the
  oracle sees every variable, in index order.
* ``get_feasible_expression`` -- the 7-variable truss feasibility CNF used in
  Sections 12.4-12.7; it has exactly one satisfying assignment.
* ``get_void_expression`` / ``decode_void_measurement`` -- coordinate encoding
  for the microstructure void search of Section 12.8.

Grover's algorithm itself is assembled in the text from Qiskit's
``grover_operator``; there is no Grover driver here.

Reference: Grover (1996), "A fast quantum mechanical algorithm for database
search"; Nielsen & Chuang (2010), Chapter 6.
"""
import re

import numpy as np

from qiskit.circuit.library import PhaseOracleGate  # noqa: F401  (re-exported for the notebook)

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

    expr = bitstring_expr.strip()

    # Single pure bitstring, e.g. "101"
    if re.fullmatch(r"[01]+", expr):
        return convert_single_bitstring(expr)

    # Boolean expression case: replace only binary tokens (e.g. 101, 001)
    # while preserving operators/parentheses.
    def repl(match):
        """
        Convert a matched binary token to a parenthesized clause.

        Regex-substitution callback: takes the whole matched bitstring token
        and returns its per-variable conjunction wrapped in parentheses so it
        can be embedded within a larger Boolean expression.

        Parameters
        ----------
        match : re.Match
            Match object whose group(0) is a binary token (e.g. '101').

        Returns
        -------
        str
            Parenthesized conjunction, e.g. '(x0 & ~x1 & x2)'.
        """
        return f"({convert_single_bitstring(match.group(0))})"

    return re.sub(r"\b[01]+\b", repl, expr)

def get_qiskit_expression(expression, n, prefix_vars=None):
    """
    Pad expression with tautologies so PhaseOracleGate sees all
    variables in the correct order.

    Parameters
    ----------
    expression   : str        Boolean expression
    n            : int        number of variables (used if prefix_vars is None)
    prefix_vars  : list[str]  explicit variable names, e.g. ['xi0','xi1','xj0','xj1']
                              If None, defaults to ['x0', 'x1', ..., 'x{n-1}']
    """
    if prefix_vars is None:
        prefix_vars = [f"x{i}" for i in range(n)]
    prefix = " & ".join(f"({v} | ~{v})" for v in prefix_vars)
    return prefix + " & " + expression


def get_feasible_expression():
    """
    Return a hard-coded satisfiable Boolean expression over 7 variables.

    Provides a fixed conjunctive-normal-form (CNF) SAT instance on variables
    x0..x6 that has exactly one feasible solution, used as a worked example
    for building a Grover PhaseOracle in this chapter.

    Returns
    -------
    str
        Boolean expression (AND of OR-clauses) in PhaseOracleGate syntax.
    """
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
    all_vars  = [f"xi{k}" for k in range(m)] + [f"xj{k}" for k in range(m)]
    return expression, n_qubits, void_coords, all_vars


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
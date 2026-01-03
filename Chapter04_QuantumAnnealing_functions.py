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
- Suresh, S. and Suresh, K., 2025. Optimal box contraction for solving linear systems via simulated and quantum annealing. Engineering Optimization, 57(9), pp.2597-2608.
- Kadowaki & Nishimori (1998): Quantum annealing in transverse Ising model
- Lucas (2014): Ising formulations of many NP problems

- PyQUBO documentation for symbolic QUBO construction
"""
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

from pyqubo import Binary, Array,Placeholder
from dimod.reference.samplers import ExactSolver
import neal
from dwave.system import LeapHybridSampler, DWaveSampler, EmbeddingComposite


import numpy as np
import matplotlib.pyplot as plt
from pyqubo import Binary, Array
from dimod.reference.samplers import ExactSolver
import neal
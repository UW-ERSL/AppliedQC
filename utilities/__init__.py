"""
Quantum Computing Textbook Utilities
=====================================

Common utilities for quantum computing examples across all chapters.

Modules
-------
quantum_utils : Qiskit quantum computing helpers
dwave_utils : D-Wave quantum annealing helpers  
plotting_utils : Visualization and plotting functions

Example Usage
-------------
>>> from utilities import quantum_utils, plotting_utils
>>> circuit = QuantumCircuit(2, 2)
>>> circuit.h(0)
>>> circuit.measure_all()
>>> counts = quantum_utils.simulateCircuit(circuit, shots=1000)
>>> plotting_utils.plot_measurement_results(counts)
"""

from . import quantum_utils
from . import dwave_utils
from . import plotting_utils

__version__ = '1.0.0'
__author__ = 'Quantum Computing Textbook'

__all__ = [
    'quantum_utils',
    'dwave_utils', 
    'plotting_utils',
]

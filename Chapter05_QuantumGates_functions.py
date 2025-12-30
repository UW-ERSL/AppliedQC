"""
Quantum Gates Simulation and Visualization
===========================================
Utilities for simulating quantum circuits and visualizing measurement outcomes.

Key Concepts:
- Quantum gates are unitary operations on qubits (reversible, preserve probability)
- Measurement collapses superposition to classical outcomes
- Circuit simulation: O(2^n) space complexity for n qubits (exponential in qubit count)

Common Single-Qubit Gates:
- X (NOT): Pauli-X, bit flip
- Y, Z: Other Pauli operators
- H (Hadamard): Creates equal superposition, basis for many quantum algorithms
- S, T: Phase gates (S = sqrt(Z), T = sqrt(S))

Common Two-Qubit Gates:
- CNOT (CX): Controlled-NOT, creates entanglement
- CZ: Controlled-Z
- SWAP: Exchanges qubit states

References:
- Nielsen & Chuang (2010): "Quantum Computation and Quantum Information"
- Qiskit documentation: https://qiskit.org/documentation/
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

import matplotlib.pyplot as plt
import numpy as np

def simulateCircuit(circuit, shots=1000):
    """
    Simulate a quantum circuit using Qiskit Aer simulator.
    
    Simulates quantum circuit evolution classically using statevector method.
    Note: Classical simulation has exponential cost O(2^n) in both time and memory,
    where n is the number of qubits. This limits practical simulation to ~30-40 qubits.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit to simulate
    shots : int, optional
        Number of measurement shots (default: 1000)
        More shots reduce statistical error: uncertainty ~ 1/sqrt(shots)
        For precise probability estimates, use shots ≥ 10000
    
    Returns
    -------
    dict
        Dictionary of measurement outcomes and their counts
        Keys are bitstrings (e.g., '01' for |01⟩), values are integer counts
    
    Complexity
    ----------
    Time: O(shots × 2^n × gates) where n = number of qubits
    Space: O(2^n) to store statevector
    
    Example
    -------
    >>> from qiskit import QuantumCircuit
    >>> circuit = QuantumCircuit(1, 1)
    >>> circuit.h(0)  # Apply Hadamard gate
    >>> circuit.measure(0, 0)
    >>> counts = simulateCircuit(circuit, shots=1000)
    >>> print(counts)
    {'0': ~500, '1': ~500}  # Equal superposition: |0⟩ + |1⟩ → 50/50 probability
    """
    backend = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(circuit, backend)
    job = backend.run(transpiled_circuit, shots=shots)
    counts = job.result().get_counts(circuit)
    return counts

def runCircuitOnIBMQuantum(circuit,nShots=1024):
	"""
	Execute circuit on IBM Quantum hardware
	=======================================
	Runs quantum circuit on actual quantum computer (not simulator).
	
	Key differences from simulation:
	- Real hardware subject to noise: decoherence, gate errors, readout errors
	- Limited connectivity: not all qubits can interact directly
	- Requires transpilation to hardware-native gates
	- Queue time depends on system availability
	
	Parameters
	----------
	circuit : QuantumCircuit
		Circuit to execute (will be transpiled for hardware)
	nShots : int
		Number of measurement repetitions (default: 1024)
		More shots reduce statistical noise but cost more compute time
	
	Returns
	-------
	dict
		Measurement counts from actual quantum hardware
		
	Note: Requires IBM Quantum account and appropriate imports:
		from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
		from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
	"""
	service = QiskitRuntimeService()
	# Select least busy backend to minimize queue time
	backend = service.least_busy(operational=True, simulator=False)
	print(f"Using backend: {backend}")
	circuit.measure_all()
	# Transpile to hardware-native gates and qubit topology
	# optimization_level=1: balance between depth reduction and compilation time
	pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
	isa_circuit = pm.run(circuit)  # ISA = Instruction Set Architecture
	sampler = Sampler(mode=backend) 
	job = sampler.run([isa_circuit],shots = nShots)
	pub_result = job.result()[0]
	return pub_result.data.meas.get_counts()

def plot_measurement_results(counts, title="Measurement Results", figsize=(10, 6)):
    """
    Create a bar plot of quantum measurement results.
    
    Visualizes probability distribution from quantum measurement outcomes.
    Bar heights represent frequency (not probability) - divide by total shots for probabilities.
    
    Parameters
    ----------
    counts : dict
        Dictionary of measurement outcomes and counts
        Keys: bitstring measurement outcomes (e.g., '00', '01', '10', '11')
        Values: integer counts from shots
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height) in inches
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
    
    Interpretation Guidelines
    ------------------------
    - Ideal quantum states: Clear bars at expected outcomes
    - Noise/errors: Unexpected bars at other outcomes
    - Superposition: Multiple bars with comparable heights
    - Statistical noise: Fluctuations scale as ~sqrt(N)/N where N = total shots
    
    Example
    -------
    >>> # Bell state: Should show only '00' and '11' with ~50/50 distribution
    >>> counts = {'00': 250, '01': 245, '10': 255, '11': 250}
    >>> fig = plot_measurement_results(counts)
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    states = list(counts.keys())
    values = list(counts.values())
    
    ax.bar(states, values, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Measurement Outcome', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Rotate x-labels if many states
    if len(states) > 8:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig
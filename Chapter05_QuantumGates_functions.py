# Functions for Chapter 5: Quantum Gates

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

import matplotlib.pyplot as plt
import numpy as np

def simulateCircuit(circuit, shots=1000):
    """
    Simulate a quantum circuit using Qiskit Aer simulator.
    
    Parameters
    ----------
    circuit : QuantumCircuit
        The quantum circuit to simulate
    shots : int, optional
        Number of measurement shots (default: 1000)
    
    Returns
    -------
    dict
        Dictionary of measurement outcomes and their counts
    
    Example
    -------
    >>> from qiskit import QuantumCircuit
    >>> circuit = QuantumCircuit(1, 1)
    >>> circuit.h(0)
    >>> circuit.measure(0, 0)
    >>> counts = simulateCircuit(circuit, shots=1000)
    >>> print(counts)
    {'0': ~500, '1': ~500}
    """
    backend = Aer.get_backend('qasm_simulator')
    transpiled_circuit = transpile(circuit, backend)
    job = backend.run(transpiled_circuit, shots=shots)
    counts = job.result().get_counts(circuit)
    return counts

def runCircuitOnIBMQuantum(circuit,nShots=1024):
	service = QiskitRuntimeService()
	backend = service.least_busy(operational=True, simulator=False)
	print(f"Using backend: {backend}")
	circuit.measure_all()
	pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
	isa_circuit = pm.run(circuit) # optimize the circuit for hardware
	sampler = Sampler(mode=backend) 
	job = sampler.run([isa_circuit],shots = nShots)
	pub_result = job.result()[0]
	return pub_result.data.meas.get_counts()

def plot_measurement_results(counts, title="Measurement Results", figsize=(10, 6)):
    """
    Create a bar plot of quantum measurement results.
    
    Parameters
    ----------
    counts : dict
        Dictionary of measurement outcomes and counts
    title : str, optional
        Plot title
    figsize : tuple, optional
        Figure size (width, height)
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    
    Example
    -------
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
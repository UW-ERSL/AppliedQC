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
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def simulateCircuit(circuit, method='matrix_product_state', shots=1000, 
                     do_transpile=True, noise_model=None):
    """
    Simulates a circuit (including MCX gates) with optional noise.
    """
    # AerSimulator supports MCX natively in most methods
    simulator = AerSimulator(method=method, noise_model=noise_model)
    
    # We must transpile if there is noise, otherwise MCX stays as one high-level block
    should_transpile = do_transpile or (noise_model is not None)
    
    if should_transpile:
        # Transpilation breaks MCX into basis gates (CX, SX, RZ, etc.)
        input_circuit = transpile(circuit, simulator)
    else:
        # High-level simulation: MCX is treated as a single large unitary matrix
        input_circuit = circuit
        
    job = simulator.run(input_circuit, shots=shots)
    return job.result().get_counts()

def analyzeCircuitForHardware(circuit, min_num_qubits=15):
    """
    Analyzes the circuit's compatibility and expected performance on hardware.
    """

    service = QiskitRuntimeService()
	# Select least busy backend to minimize queue time
    backend = service.least_busy(min_num_qubits=min_num_qubits, operational=True, simulator=False)
    print(f"Using backend: {backend}")
    # 1. Transpile to see what the hardware actually 'sees'
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(circuit)
    
    # 2. Extract key metrics
    gate_counts = isa_circuit.count_ops()
    depth = isa_circuit.depth()
    
    print(f"--- Hardware Analysis for {backend.name} ---")
    print(f"Original Gate Count: {sum(circuit.count_ops().values())}")
    print(f"Transpiled Gate Count: {sum(gate_counts.values())}")
    print(f"Circuit Depth: {depth}")
    print(f"Multi-Qubit (CX/ECR) Gates: {gate_counts.get('cx', gate_counts.get('ecr', 0))}")
    
    # 3. Warning for students
    if depth > 100:
        print("Warning: High depth. Results may be dominated by decoherence (noise).")
    
    return isa_circuit

def runCircuitOnIBMQuantum(circuit,shots=1024,min_num_qubits=15,):
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
	shots : int
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
	backend = service.least_busy(min_num_qubits=min_num_qubits, operational=True, simulator=False)
	print(f"Using backend: {backend}")
	#circuit.measure_all()
	# Transpile to hardware-native gates and qubit topology
	# optimization_level=3: aggressive optimization for depth reduction (FOR NISQ)
	pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
	isa_circuit = pm.run(circuit)  # ISA = Instruction Set Architecture
	sampler = Sampler(mode=backend) 
    # Enable measurement twirling (TREX-style mitigation)
    # This will offset any systematic measurement bias but increases shots needed
	sampler.options.twirling.enable_measure = True
	sampler.options.twirling.num_randomizations = 32  # Standard for good balance
	job = sampler.run([isa_circuit],shots = shots)
	pub_result = job.result()[0]
	return pub_result.data.c.get_counts()

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
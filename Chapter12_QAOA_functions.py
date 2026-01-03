import matplotlib.pyplot as plt
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import  Aer
from scipy.optimize import minimize
import numpy as np

def create_qaoa_circuit(gamma, beta, p=1):
    """QAOA circuit for Max-Cut on 3-node triangle"""
    qc = QuantumCircuit(3)
    
    # Initialize: equal superposition
    qc.h(range(3))
    
    # QAOA layers
    for layer in range(p):
        # Problem unitary U_P(gamma)
        # Each edge contributes exp(-i*gamma*Z_i*Z_j)
        edges = [(0,1), (1,2), (0,2)]
        for i, j in edges:
            qc.rzz(2*gamma[layer], i, j)
        
        # Mixer unitary U_M(beta)  
        for qubit in range(3):
            qc.rx(2*beta[layer], qubit)
    
    qc.measure_all()
    return qc

def compute_expectation(counts, edges):
    """Compute <H_P> from measurement counts"""
    expectation = 0.0
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Convert bitstring to array (Qiskit orders as q2 q1 q0)
        bits = [int(b) for b in bitstring[::-1]]
        
        # Compute energy: 0.5 * sum of Z_i*Z_j over edges
        energy = 0.0
        for i, j in edges:
            zi = 1 if bits[i] == 0 else -1
            zj = 1 if bits[j] == 0 else -1
            energy += 0.5 * zi * zj
        
        expectation += energy * count / total_shots
    
    return expectation

def qaoa_objective(params, edges, p, shots=1000):
    """Cost function for classical optimizer"""
    gamma = params[:p]
    beta = params[p:]
    
    qc = create_qaoa_circuit(gamma, beta, p)
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()
    
    return compute_expectation(counts, edges)



def create_random_graph(n_nodes, edge_prob=0.5):
    """Generate random graph with given edge probability"""
    edges = []
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            if np.random.rand() < edge_prob:
                edges.append((i, j))
    return edges

def estimate_circuit_quality(n_qubits, n_edges, p):
    """Estimate fidelity based on Chapter 8 analysis"""
    depth_per_layer = n_edges + n_qubits  # RZZ + RX gates
    total_depth = depth_per_layer * p
    epsilon = 0.001  # Error per gate
    
    fidelity = (1 - epsilon)**total_depth
    return fidelity, total_depth


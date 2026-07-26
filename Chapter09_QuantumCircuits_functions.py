
"""

"""
import numpy as np
import matplotlib.pyplot as plt

from qiskit import  transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler

def estimateCircuitGatesNISQ(circuit):
    """
    NISQ complexity metrics: width, depth, and size.

    Transpiles the circuit to the simulator's (continuous) basis gates ---
    the same kind of native, arbitrary-angle gate set that today's NISQ
    hardware executes directly --- and reports the metrics relevant to
    *running* the circuit on such hardware:

        * width  = number of qubits
        * depth  = length of the critical path
        * size   = total gate count (split into single-qubit and CX/ECR)

    Note: this basis is continuous, so it does NOT report a meaningful
    T-count (arbitrary-angle rotations absorb T gates). For the
    fault-tolerant T-count metric, use estimateCircuitGatesFTC().
    """

    simulator = AerSimulator()
    # Transpile to decompose MCX and adapt to simulator basis gates
    decomposedCircuit = circuit.decompose(reps = 10)
    transpiled_circuit = transpile(decomposedCircuit, simulator)

    # Extract key metrics
    gate_counts = transpiled_circuit.count_ops()
    depth = transpiled_circuit.depth()

    total_gates = sum(gate_counts.values())
    cx_gates = gate_counts.get('cx', gate_counts.get('ecr', 0))
    singleGateCount = total_gates - cx_gates

    result = {
         'num_qubits': transpiled_circuit.num_qubits,
        'single_gate_count': singleGateCount,
        'cx_gates': cx_gates,
        'total_gates': total_gates,
        'depth': depth,
        'transpiled_circuit': transpiled_circuit,
    }

    print("--- Circuit Analysis (NISQ: continuous basis) ---")
    print(f"Qubits (width):     {result['num_qubits']}")
    print(f"Depth:              {result['depth']}")
    print(f"Size (total gates): {result['total_gates']}")
    print(f"  Single-qubit:     {result['single_gate_count']}")
    print(f"  CX/ECR:           {result['cx_gates']}")

    return result


def estimateCircuitGatesFTC(circuit):
    """
    Fault-tolerant complexity metric: T-count.

    Transpiles the circuit to a discrete Clifford+T basis --- the gate set
    available on a fault-tolerant, error-corrected machine, where Clifford
    gates are cheap (transversal) and the non-Clifford T gate is the
    expensive resource (magic-state distillation). In this basis the T gate
    is atomic, so counting it is meaningful.

        * T-count  = number of T (and T-dagger) gates
        * Clifford = everything else

    Caveat: arbitrary-angle rotations (e.g. RX(0.37)) are not exactly
    expressible in Clifford+T; the transpiler approximates them
    (Solovay--Kitaev), so their contribution to the T-count is an
    approximation that grows with the accuracy demanded. The count is exact
    for circuits already built from {H, S, T, CNOT, Toffoli, ...}.
    """

    # Discrete Clifford + T basis: T is atomic here, so its count is meaningful.
    cliffordT_basis = ['h', 's', 'sdg', 'x', 'y', 'z', 'cx', 't', 'tdg']
    decomposedCircuit = circuit.decompose(reps = 10)
    transpiled_circuit = transpile(decomposedCircuit,
                                   basis_gates=cliffordT_basis,
                                   optimization_level=3)

    gate_counts = transpiled_circuit.count_ops()
    t_count = gate_counts.get('t', 0) + gate_counts.get('tdg', 0)
    total_gates = sum(gate_counts.values())
    clifford_count = total_gates - t_count

    result = {
        'num_qubits': transpiled_circuit.num_qubits,
        't_count': t_count,
        'clifford_count': clifford_count,
        'total_gates': total_gates,
        'gate_counts': dict(gate_counts),
        'transpiled_circuit': transpiled_circuit,
    }

    print("--- Circuit Analysis (Fault-tolerant: Clifford+T basis) ---")
    print(f"Qubits (width):  {result['num_qubits']}")
    print(f"T-count:         {result['t_count']}")
    print(f"Clifford gates:  {result['clifford_count']}")
    print(f"Total gates:     {result['total_gates']}")

    return result


def estimateCircuitFidelity(circuit, method='matrix_product_state', shots=1000, noise_model=None):
    """
    Analyzes the circuit's gate counts and depth after transpilation for simulation.
    """

    simulator = AerSimulator(method=method, noise_model=noise_model)
    
    # Transpile to decompose MCX and adapt to simulator basis gates
    transpiled_circuit = transpile(circuit, simulator)
    
    # Extract key metrics
    gate_counts = transpiled_circuit.count_ops()
    depth = transpiled_circuit.depth()
    
    print(f"--- Simulator Analysis ({method}) ---")
    print(f"Number of qubits: {transpiled_circuit.num_qubits}")
    print(f"Original Gate Count: {sum(circuit.count_ops().values())}")
    print(f"Transpiled Gate Count: {sum(gate_counts.values())}")
    print(f"Circuit Depth: {depth}")
    print(f"Multi-Qubit (CX/ECR) Gates: {gate_counts.get('cx', gate_counts.get('ecr', 0))}")
    total_gates = sum(gate_counts.values())
    cx_gates = gate_counts.get('cx', gate_counts.get('ecr', 0))

    if (transpiled_circuit.num_qubits > 30) and (method != 'matrix_product_state'):
        print("Warning: High qubit count with non-MPS method may lead to memory issues.")

    singleGateCount = total_gates - cx_gates
    fidelity = (1-0.001) ** singleGateCount * (1-0.003)**cx_gates
    return fidelity
"""
Circuit complexity metrics for Chapter 9, "Quantum Circuits".

Three helpers for sizing a circuit before it is run:

* ``estimateCircuitGatesNISQ`` -- width, depth and gate counts after
  transpilation to a continuous (arbitrary-angle) basis, the kind of native
  gate set today's NISQ hardware executes directly.
* ``estimateCircuitGatesFTC``  -- T-count after transpilation to a discrete
  Clifford+T basis, the resource that dominates on fault-tolerant hardware.
* ``estimateCircuitFidelity``  -- a crude gate-count-based fidelity estimate
  using the per-gate error model of Chapter 11.

All three transpile at a pinned ``optimization_level`` so that the numbers
printed in the book are reproducible across Qiskit releases; the transpiler's
default level changed from 1 to 2 in Qiskit 2.x, which silently changes every
count reported here.
"""
from qiskit import transpile
from qiskit_aer import AerSimulator

def estimateCircuitGatesNISQ(circuit, basis_gates=None, optimization_level=1):
    """
    NISQ complexity metrics: width, depth, and size.

    Transpiles the circuit to a continuous (arbitrary-angle) basis --- the
    same kind of native gate set that today's NISQ hardware executes
    directly --- and reports the metrics relevant to *running* the circuit on
    such hardware:

        * width  = number of qubits
        * depth  = length of the critical path
        * size   = total gate count (split into single-qubit and two-qubit)

    Parameters
    ----------
    circuit : QuantumCircuit
        Circuit to analyse. It is decomposed first, so high-level gates
        (MCX, controlled-U, PhaseOracleGate, ...) are unrolled.
    basis_gates : list[str], optional
        Target basis. If None (default), the simulator's own basis is used.
        Pass e.g. ``['u3', 'cx']`` to count against that gate set.
    optimization_level : int, optional
        Pinned at 1 by default. The transpiler's default moved from 1 to 2 in
        Qiskit 2.x, which changes every count below; pin it so the numbers are
        reproducible.

    Note: this basis is continuous, so it does NOT report a meaningful
    T-count (arbitrary-angle rotations absorb T gates). For the
    fault-tolerant T-count metric, use estimateCircuitGatesFTC().
    """

    # Transpile to decompose MCX and adapt to the target basis gates
    decomposedCircuit = circuit.decompose(reps = 10)
    if basis_gates is None:
        transpiled_circuit = transpile(decomposedCircuit, AerSimulator(),
                                       optimization_level=optimization_level)
    else:
        transpiled_circuit = transpile(decomposedCircuit,
                                       basis_gates=basis_gates,
                                       optimization_level=optimization_level)

    # Extract key metrics
    gate_counts = transpiled_circuit.count_ops()
    depth = transpiled_circuit.depth()

    # Count two-qubit gates by arity, not by name: cp, rzz, crx, cry, ... are
    # all two-qubit gates and were previously booked as single-qubit.
    total_gates = sum(gate_counts.values())
    cx_gates = sum(1 for instruction in transpiled_circuit.data
                   if len(instruction.qubits) == 2
                   and instruction.operation.name not in ('barrier',))
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
    print(f"  Two-qubit:        {result['cx_gates']}")

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
    # optimization_level is pinned (3) so the T-count is reproducible.
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


def estimateCircuitFidelity(circuit, method='matrix_product_state', shots=1000,
                            noise_model=None, F1=0.999, F2=0.997,
                            optimization_level=1):
    """
    Estimate circuit fidelity from transpiled gate counts.

    Applies the compounding model of Chapter 11,
    ``F_circuit = F1**n1 * F2**n2``, to the transpiled gate counts.

    Parameters
    ----------
    F1, F2 : float, optional
        Single- and two-qubit gate fidelities. Defaults 0.999 and 0.997.
    optimization_level : int, optional
        Pinned at 1; see estimateCircuitGatesNISQ.
    """

    simulator = AerSimulator(method=method, noise_model=noise_model)

    # Transpile to decompose MCX and adapt to simulator basis gates
    transpiled_circuit = transpile(circuit, simulator,
                                   optimization_level=optimization_level)
    
    # Extract key metrics
    gate_counts = transpiled_circuit.count_ops()
    depth = transpiled_circuit.depth()
    
    print(f"--- Simulator Analysis ({method}) ---")
    print(f"Number of qubits: {transpiled_circuit.num_qubits}")
    print(f"Original Gate Count: {sum(circuit.count_ops().values())}")
    print(f"Transpiled Gate Count: {sum(gate_counts.values())}")
    print(f"Circuit Depth: {depth}")
    total_gates = sum(gate_counts.values())
    cx_gates = sum(1 for instruction in transpiled_circuit.data
                   if len(instruction.qubits) == 2
                   and instruction.operation.name not in ('barrier',))
    print(f"Multi-Qubit Gates: {cx_gates}")

    if (transpiled_circuit.num_qubits > 30) and (method != 'matrix_product_state'):
        print("Warning: High qubit count with non-MPS method may lead to memory issues.")

    singleGateCount = total_gates - cx_gates
    fidelity = (F1 ** singleGateCount) * (F2 ** cx_gates)
    return fidelity
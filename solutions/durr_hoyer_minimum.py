"""
Exercise: Finding the Minimum Value — Dürr–Høyer Algorithm (Chapter 11)
=======================================================================
Find the smallest value in an unsorted table T[0..N-1] using repeated
Grover searches as a subroutine.

Algorithm (Dürr & Høyer, 1996):
  1. Pick a random index y (threshold = T[y]).
  2. Build an oracle marking all j with T[j] < T[y].
  3. Run Grover to find such a j; if found, set y ← j.
  4. Repeat until a total budget of ≈ 22.5 √N oracle calls is exhausted.

Setup:
  N = 16 (n = 4 qubits)
  T = [9, 3, 14, 1, 7, 11, 0, 5, 13, 2, 15, 8, 4, 12, 6, 10]
  True minimum: T[6] = 0

References:
  - Dürr, C. & Høyer, P. (1996). "A quantum algorithm for finding the
    minimum." arXiv:quant-ph/9607014.
  - Grover, L. K. (1997). Physical Review Letters, 79(2), 325.

Requires:
  Chapter08_QuantumGates_functions.py  (simulate_measurements)
  Chapter11_GroverAlgorithm_functions.py (bitstring_to_expression)
"""

import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseOracleGate, grover_operator
from qiskit import ClassicalRegister

from Chapter08_QuantumGates_functions import simulate_measurements
from Chapter11_GroverAlgorithm_functions import bitstring_to_expression

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Problem setup
# ═══════════════════════════════════════════════════════════════════════════
T = [9, 3, 14, 1, 7, 11, 0, 5, 13, 2, 15, 8, 4, 12, 6, 10]
N = len(T)                              # 16
n = int(np.ceil(np.log2(N)))            # 4 qubits
BUDGET = int(np.ceil(22.5 * np.sqrt(N)))  # ≈ 90 oracle calls per run

true_min_idx = int(np.argmin(T))
true_min_val = T[true_min_idx]
print(f"Table T        : {T}")
print(f"N = {N}, n = {n} qubits")
print(f"True minimum   : T[{true_min_idx}] = {true_min_val}")
print(f"Oracle budget  : {BUDGET} calls per run")


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Threshold oracle builder
# ═══════════════════════════════════════════════════════════════════════════
def build_threshold_oracle(T, threshold, n):
    """
    Build a PhaseOracleGate that marks every index j with T[j] < threshold.

    Strategy: construct a Boolean expression that is the OR of all
    bitstrings corresponding to marked indices, then feed it to
    PhaseOracleGate.  For N = 16 this is efficient.

    Returns (oracle_gate, M) where M is the number of marked items.
    """
    marked_indices = [j for j in range(len(T)) if T[j] < threshold]
    M = len(marked_indices)
    if M == 0:
        return None, 0

    # Build a Boolean expression: OR of conjunctions for each marked index
    clauses = []
    for idx in marked_indices:
        bitstring = format(idx, f'0{n}b')
        clauses.append("(" + bitstring_to_expression(bitstring) + ")")
    expression = " | ".join(clauses)

    oracle = PhaseOracleGate(expression, label="Threshold")
    return oracle, M


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Single Grover search with known M (multi-solution case, Eq. 11.8)
# ═══════════════════════════════════════════════════════════════════════════
def grover_search(oracle, n, M, N):
    """
    Run one Grover search and return the measured index (int).

    Uses the multi-solution optimal iteration count from Eq. (11.8):
        K ≈ π / (4 arcsin(√(M/N))) − 1/2

    Returns (measured_index, oracle_calls).
    """
    grover_op = grover_operator(oracle)

    # Optimal iterations for M solutions (Eq. 11.8)
    theta = math.asin(math.sqrt(M / N))
    K = max(1, math.floor(math.pi / (4 * theta) - 0.5))

    # Build circuit (same pattern as Listing 11.7)
    qc = QuantumCircuit(grover_op.num_qubits)
    qc.h(range(n))
    qc.compose(grover_op.power(K), inplace=True)

    cr = ClassicalRegister(n)
    qc.add_register(cr)
    for i in range(n):
        qc.measure(i, cr[i])

    counts = simulate_measurements(qc, shots=1)
    measured_bitstring = list(counts.keys())[0]
    measured_index = int(measured_bitstring, 2)

    return measured_index, K


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Dürr–Høyer main loop
# ═══════════════════════════════════════════════════════════════════════════
def durr_hoyer(T, n, N, budget):
    """
    Dürr–Høyer quantum minimum-finding algorithm.

    Returns (best_index, total_oracle_calls).
    """
    # Step 1: Pick a random initial guess
    y = random.randint(0, N - 1)
    total_calls = 0

    while total_calls < budget:
        threshold = T[y]

        # Step 2: Build oracle marking indices j with T[j] < threshold
        oracle, M = build_threshold_oracle(T, threshold, n)

        if M == 0:
            # Current guess is already the minimum — nothing to mark
            break

        # Step 3: Grover search for an improvement
        j, K = grover_search(oracle, n, M, N)
        total_calls += K

        # Update threshold if improvement found
        if T[j] < T[y]:
            y = j

    return y, total_calls


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Run 200 trials and collect statistics
# ═══════════════════════════════════════════════════════════════════════════
NUM_TRIALS = 200
results    = []       # returned index per trial
call_counts = []      # oracle calls per trial

random.seed(42)
for trial in range(NUM_TRIALS):
    idx, calls = durr_hoyer(T, n, N, BUDGET)
    results.append(idx)
    call_counts.append(calls)
    if (trial + 1) % 50 == 0:
        print(f"  Completed {trial+1}/{NUM_TRIALS} trials ...")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  Report statistics
# ═══════════════════════════════════════════════════════════════════════════
successes = sum(1 for r in results if r == true_min_idx)
success_prob = successes / NUM_TRIALS
avg_calls = np.mean(call_counts)

print(f"\n{'='*60}")
print(f"  Dürr–Høyer Results  ({NUM_TRIALS} trials)")
print(f"{'='*60}")
print(f"  True minimum index       : {true_min_idx}  (T[{true_min_idx}] = {true_min_val})")
print(f"  Empirical success prob.  : {successes}/{NUM_TRIALS} = {success_prob:.2%}")
print(f"  Theoretical guarantee    : >= 50%")
print(f"  Average oracle calls     : {avg_calls:.1f}  (budget = {BUDGET})")
print(f"{'='*60}")

# ═══════════════════════════════════════════════════════════════════════════
# 7.  Histogram of returned indices
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

# Count occurrences of each returned index
index_counts = {i: 0 for i in range(N)}
for r in results:
    index_counts[r] += 1

indices = list(range(N))
counts_list = [index_counts[i] for i in indices]
labels = [f"{i}\n(T={T[i]})" for i in indices]

colors = ['#2ca02c' if i == true_min_idx else 'steelblue' for i in indices]
bars = ax.bar(indices, counts_list, color=colors, edgecolor='black', alpha=0.8)

ax.set_xlabel('Index $j$ (with $T[j]$)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title(
    f"Dürr–Høyer Minimum Finding: {NUM_TRIALS} runs, "
    f"success = {success_prob:.0%}, "
    f"avg calls = {avg_calls:.1f}",
    fontsize=13, fontweight='bold'
)
ax.set_xticks(indices)
ax.set_xticklabels(labels, fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ca02c', edgecolor='black', label=f'True min (idx {true_min_idx})'),
                   Patch(facecolor='steelblue', edgecolor='black', label='Other indices')]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
fig.savefig("durr_hoyer_histogram.png", dpi=150, bbox_inches='tight')
print(f"\nHistogram saved to: durr_hoyer_histogram.png")

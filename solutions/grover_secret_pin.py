"""
Exercise: Secret PIN — Grover's Algorithm (Chapter 11)
======================================================
Find a secret 4-digit PIN (0000–9999) using Grover's algorithm.

Key observations (from Ch. 11, Eqs. 11.5–11.6):
  - Classical worst case: N = 10,000 queries
  - Grover optimal iterations: K ≈ π / (4 * arcsin(sqrt(1/N)))
    For large N this simplifies to K ≈ (π/4) * sqrt(N) ≈ 78
  - Qubits needed: ceil(log2(10,000)) = 14

References:
  - Grover, L. K. (1997). "A fast quantum mechanical algorithm for
    database search." Physical Review Letters, 79(2), 325.
  - Nielsen & Chuang (2010), Ch. 6.

Requires:
  Chapter08_QuantumGates_functions.py  (simulate_measurements, plot_measurement_results)
  Chapter11_GroverAlgorithm_functions.py (bitstring_to_expression)
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit.library import PhaseOracleGate, grover_operator
from qiskit import ClassicalRegister

# ── Use the project's own helper functions (Ch. 8 and Ch. 11) ──────────────
from Chapter08_QuantumGates_functions import simulate_measurements, plot_measurement_results
from Chapter11_GroverAlgorithm_functions import bitstring_to_expression

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Problem setup
# ═══════════════════════════════════════════════════════════════════════════
SECRET_PIN = 4971                       # Choose any PIN in 0000–9999
n          = 14                         # ceil(log2(10_000)) = 14 qubits
N          = 2**n                       # full search space = 16 384

# Convert decimal PIN → 14-bit binary string
pin_bitstring = format(SECRET_PIN, f'0{n}b')
print(f"Secret PIN       : {SECRET_PIN:04d}")
print(f"Binary encoding  : {pin_bitstring}  ({n} qubits)")

# ═══════════════════════════════════════════════════════════════════════════
# 2.  Build the phase oracle  (same pattern as Ch. 11, Listing 11.7)
# ═══════════════════════════════════════════════════════════════════════════
# bitstring_to_expression converts e.g. '01001101101011' → Boolean AND of
# x_i / ~x_i terms that PhaseOracleGate can parse.
expression = bitstring_to_expression(pin_bitstring)
oracle     = PhaseOracleGate(expression, label="PIN Oracle")
print(f"Oracle expression : {expression}")

# ═══════════════════════════════════════════════════════════════════════════
# 3.  Grover operator and optimal iteration count (Eq. 11.6)
# ═══════════════════════════════════════════════════════════════════════════
grover_op = grover_operator(oracle)

# K ≈ π / (4 θ) − 1/2,  with θ = arcsin(1/√N)   [single-solution case]
K = math.floor(math.pi / (4 * math.asin(math.sqrt(1 / N))))
print(f"Search space N   : {N}")
print(f"Optimal iterations K : {K}")

# ═══════════════════════════════════════════════════════════════════════════
# 4.  Assemble the Grover circuit  (follows Listing 11.7 exactly)
# ═══════════════════════════════════════════════════════════════════════════
qc = QuantumCircuit(grover_op.num_qubits)

# State preparation: uniform superposition on the n search qubits
qc.h(range(n))

# Apply the Grover operator K times
qc.compose(grover_op.power(K), inplace=True)

# Measurement
cr = ClassicalRegister(n)
qc.add_register(cr)
for i in range(n):
    qc.measure(i, cr[i])

# ═══════════════════════════════════════════════════════════════════════════
# 5.  Simulate and collect results
# ═══════════════════════════════════════════════════════════════════════════
shots  = 1024
counts = simulate_measurements(qc, shots=shots)

# ── Identify top measurement outcome ──────────────────────────────────────
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
top_bitstring, top_count = sorted_counts[0]
found_pin = int(top_bitstring, 2)

print(f"\n{'='*55}")
print(f" Results  ({shots} shots)")
print(f"{'='*55}")
print(f" Most frequent outcome : {top_bitstring}  (decimal {found_pin:04d})")
print(f" Counts for top result : {top_count} / {shots}  "
      f"({100*top_count/shots:.1f} %)")
print(f" Secret PIN recovered  : {'YES' if found_pin == SECRET_PIN else 'NO'}")
print(f"{'='*55}")

# Print all measured outcomes for completeness
print(f"\nAll measured outcomes:")
for bitstr, cnt in sorted_counts:
    decimal_val = int(bitstr, 2)
    marker = "  <-- SECRET PIN" if decimal_val == SECRET_PIN else ""
    print(f"  {bitstr}  (PIN {decimal_val:04d})  :  {cnt} counts{marker}")

# ═══════════════════════════════════════════════════════════════════════════
# 6.  Histogram of results
# ═══════════════════════════════════════════════════════════════════════════
fig = plot_measurement_results(
    counts,
    title=f"Grover Search for Secret PIN {SECRET_PIN:04d}  "
          f"(n={n} qubits, K={K} iterations)",
    figsize=(12, 5)
)
fig.savefig("grover_secret_pin_histogram.png", dpi=150, bbox_inches='tight')
print(f"\nHistogram saved to: grover_secret_pin_histogram.png")

"""
make_figures_ch9.py -- generate the Chapter 9 figures that are NOT in the
student notebook: exercise-statement circuits, solution circuits, and the
two solution histograms.

The 24 figures that appear in the notebook are saved by the notebook itself.
Run from the repository root:

    python make_figures_ch9.py

Every circuit below was reconstructed from the exercise text and checked
against the output state or unitary quoted in the book; the check is repeated
at the bottom of this file and printed when the script runs.

Not produced here:
  CircuitConvention.png -- a hand-drawn schematic, not a Qiskit circuit.
  ExerciseSwap.png      -- the book quotes only the final state
                           1/sqrt(2)(|100> - |101>), which several different
                           circuits produce; the intended one is not
                           determined by the text.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_histogram

from Chapter08_QuantumGates_functions import simulate_measurements

FIG_DIR = 'figs/Ch9-Circuits'
STYLE = {'name': 'bw', 'creglinecolor': '#000000'}


def save_circuit(qc, name, folder=FIG_DIR, fold=-1):
    os.makedirs(folder, exist_ok=True)
    fig = qc.draw('mpl', style=STYLE, fold=fold)
    fig.savefig(f'{folder}/{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {folder}/{name}.pdf')


def save_histogram(qc, name, shots=1000, folder=FIG_DIR):
    os.makedirs(folder, exist_ok=True)
    counts = simulate_measurements(qc, shots=shots)
    fig = plot_histogram(counts)
    fig.savefig(f'{folder}/{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {folder}/{name}.pdf   counts={counts}')


# --------------------------------------------------------------- circuits
def build():
    """Return {figure name: circuit}. Names match the .tex includes."""
    c = {}

    # H on the least and most significant qubits, X on the middle one.
    qc = QuantumCircuit(3, 3)
    qc.h(0); qc.x(1); qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    c['ExerciseHXH'] = qc

    # H (x) X (x) Y  ->  i/sqrt(2) (|011> + |111>)
    qc = QuantumCircuit(3, 3)
    qc.y(0); qc.x(1); qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])
    c['ExerciseHXYCircuit'] = qc

    # Bell state 1/sqrt(2)(|01> + |10>)
    qc = QuantumCircuit(2, 2)
    qc.x(0); qc.h(1); qc.cx(1, 0)
    qc.measure([0, 1], [0, 1])
    c['ExerciseCNotBellState'] = qc

    # Exercise-statement circuit; solution state 1/sqrt(2)(|00> + |01>)
    qc = QuantumCircuit(2)
    qc.h(0); qc.z(1); qc.cx(1, 0)
    c['ExerciseBell'] = qc

    # Exercise-statement circuit; solution state |11>
    qc = QuantumCircuit(2)
    qc.h(0); qc.x(1); qc.cx(0, 1); qc.z(1); qc.cx(1, 0); qc.h(1)
    c['Exercise2CNot'] = qc

    # H-CNOT-H;  |00> -> 1/2(|00> + |01> + |10> - |11>)
    qc = QuantumCircuit(2, 2)
    qc.h(1); qc.cx(1, 0); qc.h(1)
    qc.measure([0, 1], [0, 1])
    c['ExerciseHCNOTH'] = qc

    # CP(pi/4) with one Hadamard
    qc = QuantumCircuit(2, 2)
    qc.x(0); qc.h(1); qc.cp(np.pi / 4, 0, 1)
    qc.measure([0, 1], [0, 1])
    c['ExerciseCP'] = qc

    # Swap built from three CNOTs
    qc = QuantumCircuit(2)
    qc.cx(0, 1); qc.cx(1, 0); qc.cx(0, 1)
    c['ExerciseSwapUsingCNOT'] = qc

    # ExerciseCPH followed by a swap
    qc = QuantumCircuit(2)
    qc.h(1); qc.cp(np.pi / 2, 0, 1); qc.h(0); qc.swap(0, 1)
    c['CHSwap'] = qc

    # The 4-qubit circuit of the circuit-complexity section
    qc = QuantumCircuit(4, 4)
    qc.x(0); qc.h(1); qc.z(2)
    qc.cx(0, 1); qc.s(1); qc.cx(2, 3); qc.h(3); qc.cx(1, 3)
    c['complexCircuit'] = qc

    return c


# ------------------------------------------------------------ self-check
def verify(c):
    """Check each reconstruction against the state or unitary in the book."""
    def sv(qc):
        return Statevector(qc.remove_final_measurements(inplace=False)).data

    r2 = 1 / np.sqrt(2)
    checks = []

    s = sv(c['ExerciseHXYCircuit'])
    checks.append(('ExerciseHXYCircuit',
                   np.allclose(s[0b011], 1j * r2) and np.allclose(s[0b111], 1j * r2)))

    s = sv(c['ExerciseCNotBellState'])
    checks.append(('ExerciseCNotBellState',
                   np.allclose(s[0b01], r2) and np.allclose(s[0b10], r2)))

    s = sv(c['ExerciseBell'])
    checks.append(('ExerciseBell',
                   np.allclose(s[0b00], r2) and np.allclose(s[0b01], r2)))

    s = sv(c['Exercise2CNot'])
    checks.append(('Exercise2CNot', np.allclose(s[0b11], 1.0)))

    s = sv(c['ExerciseHCNOTH'])
    checks.append(('ExerciseHCNOTH',
                   np.allclose(s, [0.5, 0.5, 0.5, -0.5])))

    s = sv(c['ExerciseCP'])
    checks.append(('ExerciseCP',
                   np.allclose(s[0b01], r2)
                   and np.allclose(s[0b11], r2 * np.exp(1j * np.pi / 4))))

    checks.append(('ExerciseSwapUsingCNOT',
                   np.allclose(Operator(c['ExerciseSwapUsingCNOT']).data,
                               np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                                         [0, 1, 0, 0], [0, 0, 0, 1]]))))

    U_book = 0.5 * np.array([[1, 1, 1, 1], [1, 1j, -1, -1j],
                             [1, -1, 1, -1], [1, -1j, -1, 1j]])
    checks.append(('CHSwap',
                   np.allclose(Operator(c['CHSwap']).data, U_book)))

    s = sv(c['complexCircuit'])
    expect = np.zeros(16, dtype=complex)
    expect[[0b0001, 0b0011, 0b1001, 0b1011]] = [0.5, 0.5j, 0.5, 0.5j]
    checks.append(('complexCircuit', np.allclose(s, expect)))

    print('Verification against the states/unitaries quoted in the book:')
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    return ok


if __name__ == '__main__':
    circuits = build()
    if not verify(circuits):
        raise SystemExit('A reconstruction does not match the book -- stopping.')

    print('\nCircuit figures:')
    save_circuit(circuits['ExerciseHXYCircuit'], 'ExerciseHXYCircuit')
    save_circuit(circuits['ExerciseCNotBellState'], 'ExerciseCNotBellState')
    save_circuit(circuits['ExerciseBell'], 'ExerciseBell')
    save_circuit(circuits['Exercise2CNot'], 'Exercise2CNot')
    save_circuit(circuits['ExerciseHCNOTH'], 'ExerciseHCNOTH')
    save_circuit(circuits['ExerciseCP'], 'ExerciseCP')
    save_circuit(circuits['ExerciseSwapUsingCNOT'], 'ExerciseSwapUsingCNOT')
    save_circuit(circuits['CHSwap'], 'CHSwap')
    save_circuit(circuits['complexCircuit'], 'complexCircuit')

    print('\nHistograms:')
    save_histogram(circuits['ExerciseHXH'], 'ExerciseHXHHistogram')
    save_histogram(circuits['ExerciseHXYCircuit'], 'ExerciseHXYHistogram')

    print('\nDone.')
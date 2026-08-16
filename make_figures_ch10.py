"""
make_figures_ch10.py -- generate the Chapter 10 figures that are not in the
student notebook.

The four figures that appear in the notebook (HadamardTestCircuit,
InnerProduct, SwapTestExample, InnerProductEstimation) are saved by the
notebook itself.  Run from the repository root:

    python make_figures_ch10.py

Not produced here:
  SwapTest.png -- the generic swap-test schematic.  The text introduces it
                  before any listing, so the intended labelling (|phi>, |psi>
                  and whether the preparation unitaries are drawn as boxes)
                  is not determined by the chapter.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

FIG_DIR = 'figs/Ch10-Tests'
STYLE = {'name': 'bw', 'creglinecolor': '#000000'}


def save_circuit(qc, name, folder=FIG_DIR, fold=-1):
    os.makedirs(folder, exist_ok=True)
    fig = qc.draw('mpl', style=STYLE, fold=fold)
    fig.savefig(f'{folder}/{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {folder}/{name}.pdf')


def imaginary_hadamard_test():
    """The imaginary Hadamard test: the real circuit with S-dagger inserted
    on the ancilla after the first Hadamard.  This is the circuit of the
    commented-out `circuit.sdg(0)` line in the notebook."""
    zeroQubit = QuantumRegister(1, '0')
    phiQubit = QuantumRegister(1, '\u03D5')
    cl = ClassicalRegister(1, 'm')
    qc = QuantumCircuit(zeroQubit, phiQubit, cl)
    qc.h(0)
    qc.sdg(0)                                     # <-- the only difference
    UMatrix = 1 / np.sqrt(2) * np.array([[1, 1], [1j, -1j]])
    qc.append(UnitaryGate(UMatrix, 'U').control(1), [0, 1])
    qc.h(0)
    qc.measure(0, 0)
    return qc


def verify():
    """Check that the circuit does estimate Im<phi|U|phi> via 2P(0)-1.

    With phi = |0> and U as above, <0|U|0> = 1/sqrt(2), which is real, so the
    imaginary part is 0 and P(0) should be 1/2.  A second check uses
    U = S (phase i), for which <0|S|0> = 1 and Im = 0, and U = the Y gate
    conjugated so that the expectation is purely imaginary.
    """
    def P0(U, prep=None):
        qc = QuantumCircuit(2)
        if prep is not None:
            qc.append(prep, [1])
        qc.h(0)
        qc.sdg(0)
        qc.append(UnitaryGate(U, 'U').control(1), [0, 1])
        qc.h(0)
        s = Statevector(qc).data.reshape(2, 2)     # index [phi, ancilla]
        return float(np.sum(np.abs(s[:, 0]) ** 2))

    checks = []
    U1 = 1 / np.sqrt(2) * np.array([[1, 1], [1j, -1j]])
    checks.append(('Im<0|U|0> = 0 for the book U',
                   abs((2 * P0(U1) - 1) - np.imag(U1[0, 0])) < 1e-9))

    # A case with a genuinely imaginary expectation: U = diag(exp(i pi/3), 1)
    th = np.pi / 3
    U2 = np.diag([np.exp(1j * th), 1.0])
    checks.append((f'Im<0|U|0> = sin(pi/3) = {np.sin(th):.4f}',
                   abs((2 * P0(U2) - 1) - np.sin(th)) < 1e-9))

    print('Verification of the imaginary Hadamard test:')
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= passed
    return ok


if __name__ == '__main__':
    if not verify():
        raise SystemExit('Imaginary Hadamard test does not check out -- stopping.')
    print('\nCircuit figures:')
    save_circuit(imaginary_hadamard_test(), 'HadamardTestImaginary')
    print('\nDone. ')

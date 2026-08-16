"""
make_figures_ch8.py -- generate the Chapter 8 figures that are NOT in the
student notebook, i.e. the exercise and solution circuits plus the
measurement-error plot.

The six circuits that appear in the notebook are saved by the notebook itself.
Run this script from the repository root:

    python make_figures_ch8.py

Output: PDF (vector) into figs/Ch8-Gates/, ready to upload to Overleaf.

Only 'figs/Ch8-Gates/QuantumGate.png' is not produced here -- it is a
hand-drawn schematic, not a Qiskit circuit.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')                      # no display needed
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

FIG_DIR = 'figs/Ch8-Gates'


def save_circuit(qc, name, folder=FIG_DIR):
    """Draw a circuit in black and white and save it as PDF."""
    os.makedirs(folder, exist_ok=True)
    fig = qc.draw('mpl', style={'name': 'bw', 'creglinecolor': '#000000'}, fold=-1)
    fig.savefig(f'{folder}/{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {folder}/{name}.pdf')


def save_plot(fig, name, folder=FIG_DIR):
    os.makedirs(folder, exist_ok=True)
    fig.savefig(f'{folder}/{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {folder}/{name}.pdf')


# ---------------------------------------------------------------- circuits
def circuit_figures():
    print('Circuit figures:')

    # Solution to the RX state-preparation exercise (Sec 8.7)
    qc = QuantumCircuit(1, 1)
    theta = 2 * np.arctan2(-1 / 3, np.sqrt(8) / 3)
    qc.rx(theta, 0)
    qc.x(0)
    qc.measure(0, 0)
    save_circuit(qc, 'StatePreparationRX')

    # Solution to the RX-then-RY exercise
    qc = QuantumCircuit(1, 1)
    qc.rx(2 * np.pi / 3, 0)
    qc.ry(np.pi / 3, 0)
    qc.measure(0, 0)
    save_circuit(qc, 'ExerciseRxRyCircuit')

    # Exercise statement: two Hadamards in sequence
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.h(0)
    qc.measure(0, 0)
    save_circuit(qc, 'ExerciseDoubleHadamard')

    # Solution to Exercise ex:HSTH -- gates drawn in application order
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.t(0)
    qc.s(0)
    qc.h(0)
    qc.measure(0, 0)
    save_circuit(qc, 'ExerciseHTSHCircuit')


# ------------------------------------------------------- measurement error
def measurement_error_plot(n_trials=100):
    """Average error in the measured probability of 0 versus number of shots,
    for the RX(pi/3) circuit whose exact P(0) = 3/4.

    Note: the runs are seeded consecutively (0, 1, 2, ...) so the figure is
    reproducible.  Do NOT draw the seeds from a single random stream -- doing
    so can give a correlated set of streams and collapse the apparent error at
    large shot counts, destroying the 1/sqrt(N) trend this figure is about.
    """
    print('Measurement-error plot:')
    p_exact = 0.75

    qc = QuantumCircuit(1, 1)
    qc.rx(np.pi / 3, 0)
    qc.measure(0, 0)

    sim = AerSimulator()
    tqc = transpile(qc, sim)

    shot_counts = [100, 1000, 10000, 100000]
    avg_err = []
    for shots in shot_counts:
        errs = [abs(sim.run(tqc, shots=shots, seed_simulator=t
                            ).result().get_counts().get('0', 0) / shots - p_exact)
                for t in range(n_trials)]
        avg_err.append(np.mean(errs))
        print(f'    shots={shots:>7,}  average error={avg_err[-1]:.5f}')

    slope = np.polyfit(np.log10(shot_counts), np.log10(avg_err), 1)[0]
    print(f'    log-log slope = {slope:.3f}  (theory -0.5)')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(shot_counts, avg_err, 'o-', color='black',
              label='measured average error')
    ref = avg_err[0] * np.sqrt(shot_counts[0] / np.array(shot_counts, float))
    ax.loglog(shot_counts, ref, '--', color='gray',
              label=r'$1/\sqrt{N}$ reference')
    ax.set_xlabel('Number of shots $N$')
    ax.set_ylabel('Average error in $P(0)$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    save_plot(fig, 'measurementError')


if __name__ == '__main__':
    circuit_figures()
    measurement_error_plot()
    print('\nDone.')
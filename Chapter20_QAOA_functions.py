import numpy as np
import time
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import numpy as np
import time
from typing import Dict, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import Parameter
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

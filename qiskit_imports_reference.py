
## This is a just a  reference file to keep track of Qiskit imports for cut-and-paste purposes. 
# It is not meant to be run as a script.


from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info import Operator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import  XGate, YGate, ZGate, RYGate, UCGate
from qiskit.circuit.library import StatePreparation
from qiskit.circuit.library import DiagonalGate
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit.circuit.library import  MCXGate
from qiskit.circuit.library import  PhaseOracle
from qiskit.circuit.library import PhaseOracleGate
from qiskit.circuit.library import grover_operator
from qiskit.circuit.library import QFT
from qiskit.circuit.library import phase_estimation
from qiskit.circuit.library import HamiltonianGate
from qiskit.circuit.library import GroverOperator
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.library.standard_gates import PhaseGate
from qiskit.circuit.library import real_amplitudes 
from qiskit.visualization import plot_histogram
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit_aer import Aer
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import Estimator
from qiskit_ibm_runtime import EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise import depolarizing_error
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import StatevectorSampler 

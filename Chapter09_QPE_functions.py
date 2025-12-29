# Functions used in Chappter 8 on QFT

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate
from Chapter05_QuantumGates_functions import simulateCircuit
from qiskit.circuit.library import QFT, phase_estimation, HamiltonianGate

def myQPESingleBit(A,v,lambdaUpper,nShots=1000):
	n = int(np.log2(v.shape[0]))
	circuit = QuantumCircuit(n+1,1)
	circuit.h(0)
	circuit.prepare_state(Statevector(v),[*range(1, n+1)],'v')
	t = -2*np.pi/lambdaUpper #Note negative
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	UControl = U_A.control(1) # only 1 control qubit
	circuit.append(UControl,[*range(0, n+1)])
	circuit.h(0)
	circuit.measure([0], [0]) 
	counts = simulateCircuit(circuit,nShots)
	print('Counts:',counts)
	probabilities = np.array([])
	thetaEstimates = np.array([])
	for key in counts:
		probabilities = np.append(probabilities,counts[key]/nShots)
		thetaEstimates = np.append(thetaEstimates,int(key, 2)/(2))
	return [thetaEstimates,probabilities]


def myQPEMultiBit(A,v,lambdaUpper,m,nShots=1000):
	N = v.shape[0]
	n = int(np.log2(N))
	phase_qubits = QuantumRegister(m, '\u03B8')
	input_qubits = QuantumRegister(n, 'v')
	phase_measurements = ClassicalRegister(m, '\u0398')
	circuit = QuantumCircuit(phase_qubits,input_qubits,phase_measurements)
	for i in range(m):
		circuit.h(i)
	circuit.prepare_state(Statevector(v),[*range(m, n+m)],'v')
	t = -2*np.pi/lambdaUpper #Note negative
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	U_A._name = 'UA'
	for i in range(m):
		U_A_pow = U_A.power(2**i) 
		UControl = U_A_pow.control(1) # only 1 control qubit
		circuit.append(UControl,[i,*range(m, n+m)])
	iqft = QFTGate(num_qubits=m).inverse()
	iqft._name = 'IQFT'
	circuit.append(iqft, [*range(0,m)])
	circuit.measure([*range(0,m)], [*range(0,m)]) 
	circuit.draw('mpl') 
	counts = simulateCircuit(circuit,nShots)
	print(counts)
	countsSorted = {k: v for k, v in sorted(counts.items(), 
										 key=lambda item: item[1],
										 reverse=True)}
	probabilities = np.array([])
	thetaEstimates = np.array([])
	for key in countsSorted:
		probabilities = np.append(probabilities,countsSorted[key]/nShots)
		thetaEstimates = np.append(thetaEstimates,int(key, 2)/(2**m))
	return [thetaEstimates,probabilities]


def QiskitQPEWrapper(A,v,lambdaUpper,m,nShots=1000):
	N = v.shape[0]
	n = int(np.log2(N))
	print(n,m)
	phase_qubits = QuantumRegister(m, '\u03B8')
	input_qubits = QuantumRegister(n, 'v')
	phase_measurements = ClassicalRegister(m, '\u0398')
	circuit = QuantumCircuit(phase_qubits,input_qubits,phase_measurements)
	circuit.prepare_state(Statevector(v),[*range(m, n+m)],'v')
	
	t = -2*np.pi/lambdaUpper #Note negative
	U_A = HamiltonianGate(A, time=t,label = 'UA')
	U_A._name = 'UA'
	QPE = phase_estimation(m,unitary=U_A)
	circuit.append(QPE, [*range(n+m)])
	circuit.measure( [*range(0, m)],[*range(m-1,-1,-1)]) 
	counts = simulateCircuit(circuit,nShots)
	
	countsSorted = {k: v for k, v in sorted(counts.items(), 
										 key=lambda item: item[1],
										 reverse=True)}
	probabilities = np.array([])
	thetaEstimates = np.array([])
	for key in countsSorted:
		probabilities = np.append(probabilities,countsSorted[key]/nShots)
		thetaEstimates = np.append(thetaEstimates,int(key, 2)/(2**m))
	return [thetaEstimates,probabilities]
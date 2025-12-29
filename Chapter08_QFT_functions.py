# Functions used in Chappter 8 on QFT

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate
from Chapter05_QuantumGates_functions import simulateCircuit

def trignometricSignal(t,c = [],s = []):
	signal = sum(ck * np.cos(k * 2 * np.pi * t) for k, ck in enumerate(c)) + \
		sum(sk * np.sin(k * 2 * np.pi * t) for k, sk in enumerate(s))
	return signal

def createDFTMatrix(M):
    omega = np.exp(-2j * np.pi / M)
    i, j = np.meshgrid(np.arange(M), np.arange(M))
    return omega ** (i * j)

def processDFTResult(phi):
	M = len(phi)
	c = np.zeros(int(M/2))
	s = np.zeros(int(M/2))
	c[0] =  phi[0].real # constant
	c[1:] = (phi[1:int(M/2)]+phi[M-1:int(M/2):-1]).real; # cosine terms
	s =  (phi[M-1:int(M/2):-1] - phi[1:int(M/2)]).imag; # sine terms
	s = np.insert(s, 0,0)
	return [c,s]


def plotDFTResult(c,s,M):
	plt.figure()
	plt.bar(list(range(0,int(M/2))),c, label =r"$c_k$")
	plt.bar(list(range(0,int(M/2))),s, label =r"$s_k$")
	plt.legend( fontsize=14)
	plt.axhline(0, color='black')
	plt.xlabel('Frequency', fontsize=14)
	plt.ylabel('Amplitude', fontsize=14)

	plt.title('DFT Result', fontsize=16)
	plt.grid(visible=True)
	plt.show()
	
def QFTSignalProcessing(y,shots=1000):
	M = len(y) # length of signal
	m = int(np.log2(M)) # number of qubits
	print('Number of qubits:',m)
	circuit = QuantumCircuit(m, m)  
	q = Statevector(y/np.linalg.norm(y)) 
	circuit.prepare_state(q,list(range(m)),'Prepare q')
	qftCircuit = QFTGate(num_qubits=m)
	circuit.append(qftCircuit, qargs=list(range(m)))
	circuit.measure(list(range(m)),list(range(m))) 
	counts = simulateCircuit(circuit,shots=shots)
		
	return counts

def processQFTResult(M,counts,shots):
	phi = np.zeros(M)
	for i in counts:
		freq = int(i, 2)
		phi[freq] = np.sqrt(counts[i]/shots)
	ampl = (phi[1:int(M/2)])+(phi[M-1:int(M/2):-1]);
	ampl = np.insert(ampl, 0,phi[0].real)
	return ampl
	
def myQFT(m): # m is the # of qubits
    q = QuantumRegister(m, 'q')
    c = ClassicalRegister(m,'c')
    circuit = QuantumCircuit(q,c)
    for k in range(m):
        kk = m - k
        circuit.h(q[kk-1])
        circuit.barrier()
        for i in reversed(range(kk-1)):
            circuit.cp(2*np.pi/2**(kk-i),q[i], q[kk-1])
      
    circuit.barrier()  
    for i in range(m//2):
        circuit.swap(q[i], q[m-i-1])
    return circuit

def createQFTMatrix(M):
    omega = np.exp(1j*(2*np.pi/M))
    i, j = np.meshgrid(np.arange(M), np.arange(M))
    QFTMatrix = omega ** (i * j)/np.sqrt(M)
    return QFTMatrix
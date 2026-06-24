"""1D Poisson (Dirichlet) block encoding, Sünderhauf-style PREP/SELECT/UNPREP.
SELECT = QFT/Draper constant-adders; boundary flags = n-control 'is-boundary'
indicators (one clean ancilla z, one recursion-synth ancilla g), uncomputed.
Verifies (<0|_anc x I)U(|0>_anc x I)*alpha == A and reports transpiled cost."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation, QFT, PhaseGate
from qiskit.quantum_info import Operator

ALPHA = 4.0
def _prep(): return StatePreparation([np.sqrt(2/ALPHA),np.sqrt(1/ALPHA),np.sqrt(1/ALPHA),0.0])

def build(n, flags='recursion'):
    """flags: 'recursion' (ancilla-assisted) or 'noancilla'."""
    N=2**n
    s=QuantumRegister(n,"s"); c=QuantumRegister(2,"c"); b=QuantumRegister(1,"b")
    regs=[s,c,b]
    if flags=='recursion':
        z=QuantumRegister(1,"z"); g=QuantumRegister(1,"g"); regs+=[z,g]
    qc=QuantumCircuit(*regs)
    P=_prep(); qc.append(P,[c[0],c[1]]); qc.z(c[0]); qc.z(c[1])

    if flags=='noancilla':
        ctrl=[c[0],c[1]]+list(s)
        qc.mcx(ctrl,b[0],ctrl_state="0"*n+"01")
        qc.mcx(ctrl,b[0],ctrl_state="1"*n+"10")
    else:
        def isall(one):
            if not one: qc.x(s)
            qc.mcx(list(s), z[0], ancilla_qubits=[g[0]], mode='recursion')
            if not one: qc.x(s)
        isall(False); qc.x(c[1]); qc.mcx([c[0],c[1],z[0]],b[0]); qc.x(c[1]); isall(False)
        isall(True);  qc.x(c[0]); qc.mcx([c[0],c[1],z[0]],b[0]); qc.x(c[0]); isall(True)

    qft=QFT(n,do_swaps=True); qc.append(qft.to_gate(),list(s))
    for q in range(n):
        ang=2*np.pi*(2**q)/N
        qc.append(PhaseGate(+ang).control(2,ctrl_state="10"),[c[0],c[1],s[q]])
        qc.append(PhaseGate(-ang).control(2,ctrl_state="01"),[c[0],c[1],s[q]])
    qc.append(qft.inverse().to_gate(),list(s)); qc.append(P.inverse(),[c[0],c[1]])
    return qc, N

def A_mat(N): return np.diag([2.0]*N)+np.diag([-1.0]*(N-1),1)+np.diag([-1.0]*(N-1),-1)

def verify(n, flags):
    qc,N=build(n,flags); U=Operator(qc).data; B=U[:N,:N]
    err=np.linalg.norm((B*ALPHA).real-A_mat(N)); ue=np.linalg.norm(U.conj().T@U-np.eye(U.shape[0]))
    assert err<1e-9 and ue<1e-9, f"FAIL n={n} flags={flags} err={err}"
    return err, ue

if __name__=="__main__":
    print("--- correctness ---")
    for fl in ['noancilla','recursion']:
        for n in [2,3,4]:
            err,ue=verify(n,fl); print(f"  {fl:10s} n={n} N={2**n:>2}: alpha=4  ||a*B-A||={err:.1e}  VERIFIED")

    print("\n--- transpiled cost vs n=log2(N)  (basis u,cx) ---")
    def metr(qc):
        t=transpile(qc,basis_gates=['u','cx'],optimization_level=1); return t.depth(),t.count_ops().get('cx',0)
    print(f"{'n':>2}{'N':>6}{'noanc cx':>10}{'anc cx':>9}{'anc depth':>11}{'CX/qubit':>10}")
    prev=None
    for n in [3,4,5,6,7,8]:
        _,na=metr(build(n,'noancilla')[0]); d,a=metr(build(n,'recursion')[0])
        slope = "" if prev is None else f"{a-prev:>+5d}"
        print(f"{n:>2}{2**n:>6}{na:>10}{a:>9}{d:>11}{slope:>10}")
        prev=a
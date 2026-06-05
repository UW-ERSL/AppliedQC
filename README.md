# Applied Quantum Computing for Engineers
### From Theory to Implementation

**Krishnan Suresh** | University of Wisconsin–Madison  
*First Edition, 2026*

---

This repository contains the companion code for the textbook  
**Applied Quantum Computing for Engineers: From Theory to Implementation**.

Each chapter includes a Jupyter notebook for interactive exploration and a Python module of reusable functions. All code is written in Python using [Qiskit](https://qiskit.org/) (gate-based quantum computing) and [D-Wave Ocean](https://docs.ocean.dwavesys.com/) (quantum annealing), and runs on both simulators and real quantum hardware.

---

## Repository Structure

Each chapter provides two files:

| File | Purpose |
|---|---|
| `ChapterXX_Topic_notebook.ipynb` | Step-by-step interactive walkthrough with explanations and outputs |
| `ChapterXX_Topic_functions.py` | Clean, importable Python functions for reuse in your own projects |

A `solutions/` folder contains worked solutions to the end-of-chapter problems.  
`qiskit_imports_reference.py` is a handy reference for common Qiskit imports used throughout the book.

---

## Chapter Guide

| Chapter | Topic | Platform |
|---|---|---|
| 2 | QC Software — Installation & Setup | Qiskit / D-Wave |
| 3 | Engineering Problems — Poisson, Truss, Plane Stress | Classical / Setup |
| 4 | Essential Math for QC | — |
| 5 | Quantum Annealing (QA) & QUBO | D-Wave |
| 6 | Real Number Encoding | D-Wave |
| 7 | Qubits | Qiskit |
| 8 | Quantum Gates | Qiskit / IBM Quantum |
| 9 | Quantum Circuits | Qiskit |
| 10 | Quantum Tests | Qiskit |
| 11 | Quantum Noise & Error Mitigation | Qiskit |
| 12 | Grover's Algorithm | Qiskit |
| 13 | Vector Encoding | Qiskit |
| 14 | Matrix Encoding (LCU, Pauli Expansion) | Qiskit |
| 15 | Amplitude Estimation (IQAE) | Qiskit |
| 16 | Quantum Fourier Transform (QFT) | Qiskit |
| 17 | Quantum Phase Estimation (QPE) | Qiskit |
| 18 | HHL Algorithm | Qiskit |
| 19 | Quantum Singular Value Transformation (QSVT) | Qiskit |
| 20 | QAOA | Qiskit |
| 21 | Variational Quantum Linear Solver (VQLS) | Qiskit |

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/UW-ERSL/AppliedQC.git
cd AppliedQC
```

### 2. Set up your environment

The book uses **Anaconda** for environment management and **VS Code** as the recommended editor (see Chapter 2 for detailed setup instructions).

Install the core dependencies:
```bash
pip install qiskit qiskit-ibm-runtime qiskit-aer
pip install dwave-ocean-sdk pyqubo dimod
pip install numpy scipy matplotlib jupyter
```

Or create a dedicated conda environment:
```bash
conda create -n appliedqc python=3.11
conda activate appliedqc
pip install qiskit qiskit-ibm-runtime qiskit-aer dwave-ocean-sdk pyqubo dimod numpy scipy matplotlib jupyter
```

### 3. Launch the notebooks
```bash
jupyter notebook
```

---

## Running on Real Quantum Hardware

**Gate-based (IBM Quantum):** Chapter 8 walks through setting up your IBM Quantum account and submitting circuits to real hardware via `qiskit-ibm-runtime`.

**Quantum Annealing (D-Wave):** Chapter 5 covers D-Wave account setup and accessing the Advantage system via the Ocean SDK.

---

## Book Description

This book is a practical, implementation-focused guide to quantum computing for engineers and applied scientists. It covers two major quantum computing paradigms — **quantum annealing** (D-Wave) and **gate-based quantum computing** (IBM Quantum / Qiskit) — within a single unified framework, grounded throughout in real engineering applications.

**Topics covered include:**
- Engineering problem formulation (Poisson equation, truss analysis, topology optimization)
- Quantum annealing and QUBO problem construction
- Gate-based quantum circuits, noise, and error mitigation
- Key algorithms: Grover, QFT, QPE, HHL, QSVT, QAOA, VQLS
- Data encoding strategies for vectors and matrices
- Running experiments on real quantum hardware

The book is designed for engineers and graduate students with a background in linear algebra and scientific computing but no prior knowledge of quantum mechanics.

---

## Citation

If you use this code in your research or teaching, please cite:

```
@book{suresh2026appliedqc,
  title     = {Applied Quantum Computing for Engineers: From Theory to Implementation},
  author    = {Suresh, Krishnan},
  year      = {2026},
  edition   = {1},
  publisher = {},
  url       = {https://github.com/UW-ERSL/AppliedQC}
}
```

---

## License

© 2026 Krishnan Suresh. All rights reserved.  
Code in this repository is provided for educational use in conjunction with the textbook.

---

*University of Wisconsin–Madison — Engineering Research & Simulation Lab (ERSL)*

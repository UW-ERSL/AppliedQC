# Installing and running the code

*Applied Quantum Computing for Engineers: Linear Systems and Optimization*
Krishnan Suresh, University of Wisconsin–Madison

There are two ways to run the notebooks in this repository. Both work; they differ
in what you install and in one detail of reproducibility, noted at the end.

| | Option A — your own machine | Option B — Google Colab |
|---|---|---|
| Install | conda environment, once | nothing |
| Python | 3.11 | 3.12 (Colab's) |
| Per-session cost | none | ~1–2 min per notebook |
| Files you keep | yes | no, unless you save a copy |

---

## Option A — local installation

This is the environment described in Chapter 2 of the book, and the one used to
produce the printed figures and transcripts.

```
conda create -n quantum python=3.11
conda activate quantum
python -m pip install ipykernel
python -m ipykernel install --user --name quantum
git clone https://github.com/UW-ERSL/AppliedQC
cd AppliedQC
python -m pip install -r requirements.txt
```

Open a notebook in VS Code and confirm the kernel selector reads `quantum`.

**`python=3.11` is deliberate.** It is what pins SciPy to 1.17.1, the version behind
every number printed in the book. A 3.12 environment also works, but installs SciPy
1.18.0 instead — see *Reproducibility* below.

**If you prefer the classic Jupyter interface** to VS Code, install
`requirements-jupyter.txt` instead of `requirements.txt`. It includes everything in
`requirements.txt` plus `jupyter`, `notebook` and `ipykernel`, so run only that one.

---

## Option B — Google Colab

Nothing to install. Open any chapter notebook directly from GitHub:

**File → Open notebook → GitHub tab**, enter `UW-ERSL/AppliedQC`, and choose a chapter.

Every notebook begins with a setup cell. Run it.

1. It clones this repository and installs the packages. About a minute or two.
2. It then restarts the session by itself and tells you so.
3. When the restart finishes, **run the same cell again.** It will be quick and print
   `Ready.` with the version numbers.
4. Work through the chapter as usual — or choose **Runtime → Run all**, which repeats
   step 3 and runs every cell. Run all is convenient, but some chapters take a while.

The restart is required because Colab loads its own NumPy before your first cell runs,
so the version this book needs only becomes active after the kernel restarts.

### What normal looks like

Three things appear on a healthy first run and none of them indicate a problem.

- **A red block from pip mentioning `numba` and `numpy`.** That is Colab's own numba
  package objecting to the NumPy version this book pins. Nothing here uses numba.
- **The session restarting itself.** Expected, and announced by the cell before it happens.
- **A coloured underline** under the `ChapterNN_..._functions` import. Colab's editor
  checks imports before any code runs, so it cannot know the setup cell put the
  repository on the path. The import works; your output proves it.

### Things to know about Colab

- **Every notebook gets its own machine.** Opening Chapter 9 after Chapter 8 means
  installing again. Nothing carries over.
- **Sessions are temporary.** Roughly 90 minutes idle, 12 hours maximum. Everything
  installed is discarded.
- **Notebooks opened from GitHub are read-only.** To keep your edits, use
  **File → Save a copy in Drive**.
- **Never type an IBM Quantum token into a cell** — notebooks save their own output. Use:

  ```python
  from getpass import getpass
  token = getpass("IBM Quantum token: ")
  ```

---

## Expected versions

After setup completes, the notebooks report:

| | Local (Python 3.11) | Colab (Python 3.12) |
|---|---|---|
| numpy | 2.4.6 | 2.4.6 |
| scipy | **1.17.1** | **1.18.0** |
| qiskit | 2.4.2 | 2.4.2 |

SciPy differs by necessity: `openjij` requires SciPy < 1.18 on Python 3.11 and ≥ 1.18 on
3.12, and SciPy 1.18 itself requires Python ≥ 3.12. No single version satisfies both, so
`requirements.txt` selects by interpreter. Do not "simplify" those two lines.

---

## Troubleshooting

**`ResolutionImpossible` during install.** Your `requirements.txt` predates the SciPy
version markers. Pull the latest `main`.

**`ImportError: cannot import name '_center' from 'numpy._core.umath'`**, or any import
error deep inside SciPy or NumPy. The session was not restarted after installing.
**Runtime → Restart session**, then run the setup cell again.

**A conflict naming `qiskit[visualization]` and `qiskit ... (Installed)`.** pip is
resolving against packages already present from an earlier run.
**Runtime → Disconnect and delete runtime**, then start over.

**`ModuleNotFoundError: ChapterNN_..._functions`.** The setup cell did not run in *this*
notebook. Each notebook has its own machine; run its first cell.

**Setup says restarting did not fix NumPy.** The machine is in a mixed state.
**Runtime → Disconnect and delete runtime** and open the notebook again.

**Anything on your own machine.** Check your versions against the table above before
assuming the book is wrong. Report what you find using the link at the top of the
README, and say which of the two environments you were using.

---

## Reproducibility

The printed figures and transcripts were produced in the **local Python 3.11**
environment, with SciPy 1.17.1. That is the reference environment.

Colab runs SciPy 1.18.0. The optimisation routines the book uses are unchanged between
those releases — COBYLA in particular produces identical results — but numbers that
depend on the underlying linear-algebra library can differ in the last few digits on any
machine, including between two local installations. If a printed value and your output
disagree beyond that, it is worth reporting.

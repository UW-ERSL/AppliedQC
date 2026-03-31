# Solutions Manual - Instructor Guide

This folder contains **solutions to textbook exercises** organized by chapter.

---

## 📁 Solutions Structure

```
solutions/
├── README_SOLUTIONS.md          ← This file
├── Chapter03_Solutions.ipynb    ← Quantum Annealing exercises
├── Chapter04_Solutions.ipynb    ← Qubits exercises
├── Chapter05_Solutions.ipynb    ← Quantum Gates exercises
├── Chapter06_Solutions.ipynb    ← Quantum Circuits exercises
├── Chapter08_Solutions.ipynb    ← QFT exercises
├── Chapter09_Solutions.ipynb    ← QPE exercises
└── Chapter11_Solutions.ipynb    ← HHL exercises
```

---

## 🔑 Key Features

### **1. Solutions Use the Same Utilities**

All solution notebooks import from `utilities/` - the **same shared code** used in main textbook notebooks:

```python
# Every solutions notebook has this setup cell
from quantum_utils import simulateCircuit, get_statevector
from plotting_utils import plot_measurement_results
```

**Benefits:**
- ✅ No code duplication
- ✅ Solutions use identical helper functions as main text
- ✅ Updates to utilities automatically propagate to solutions
- ✅ Students learn proper code organization

### **2. Organized Per-Chapter**

Each solutions notebook corresponds to **one textbook chapter**, matching the main notebooks structure:

| Solutions File | Main Notebook | Textbook Chapter |
|---------------|---------------|------------------|
| `Chapter03_Solutions.ipynb` | `Chapter03_QuantumAnnealing.ipynb` | Chapter 3 |
| `Chapter05_Solutions.ipynb` | `Chapter05_QuantumGates.ipynb` | Chapter 5 |
| `Chapter11_Solutions.ipynb` | `Chapter11_HHL.ipynb` | Chapter 11 |

### **3. Work-in-Progress Friendly**

Some chapters have incomplete solutions (marked as WIP). This is normal for a textbook in development.

**How to handle:**
- Placeholder cells indicate "Solution in development"
- Can add solutions incrementally across editions
- Students get hints even if full solutions aren't ready

---

## 👨‍🏫 For Instructors

### **Distribution Options**

**Option 1: Restricted Access** (Recommended)
- Keep solutions in instructor-only repository
- Distribute only to teaching assistants
- Share selectively with students who need extra help

**Option 2: Delayed Release**
- Release solutions after assignment deadlines
- Post solutions after exams
- Gradual release throughout semester

**Option 3: Partial Solutions**
- Give students hints (markdown cells only)
- Hide code cells until needed
- Provide worked examples for half the problems

### **Customization for Your Course**

Each solutions notebook is **independent**, so you can:

1. **Add your own exercises**
   ```python
   # Add a new exercise cell
   """
   ## Exercise 5.X - Your Custom Exercise
   
   [Problem description]
   """
   ```

2. **Remove solutions** (convert to exercises)
   - Delete code cells, keep problem descriptions
   - Give as homework assignments

3. **Modify difficulty**
   - Add hints to markdown cells
   - Break complex solutions into steps
   - Add additional test cases

4. **Create variants**
   - Change parameters (different matrices, angles, etc.)
   - Use different quantum gates
   - Extend to more qubits

---

## 🎓 For Students (When Appropriate)

If you choose to share solutions with students:

### **How to Use Solutions Effectively**

**❌ Don't:**
- Look at solutions before attempting problems
- Copy code without understanding
- Use solutions as your only learning resource

**✅ Do:**
- Try the problem yourself first (at least 15-30 minutes)
- Use solutions to check your approach
- Study solution techniques, then re-solve on your own
- Compare your solution with the provided one
- Ask yourself: "Why is this solution better/different?"

### **Setup**

Solutions use the same environment as main notebooks:

```bash
# From the solutions/ folder
jupyter notebook Chapter05_Solutions.ipynb

# The setup cell will automatically import utilities
```

If you get import errors, make sure you're running from the `solutions/` folder and `utilities/` is in the parent directory.

---

## 🔧 Technical Details

### **Import Strategy**

Solutions import from `utilities/` using relative paths:

```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent / 'utilities'))
```

This works because directory structure is:
```
quantum-computing-book/
├── solutions/         ← You are here
└── utilities/         ← One level up, then into utilities
```

### **Functions from Main Text**

**Q: What if a solution needs a function from the main chapter notebook?**

**A: Three approaches:**

1. **Best:** If the function is used in solutions, it should be in `utilities/`
   - Example: `simulateCircuit()` is in `quantum_utils.py`
   
2. **Simple:** Copy the function definition into the solutions notebook
   - For chapter-specific helpers used only in that chapter's solutions
   
3. **Advanced:** Import the main notebook (requires extra packages)
   - Generally not recommended; use approach 1 or 2

---

## 📊 Solutions Coverage

| Chapter | Solutions Notebook | Status | Cell Count |
|---------|-------------------|--------|------------|
| 3 | `Chapter03_Solutions.ipynb` | ✅ Complete | 3 |
| 4 | `Chapter04_Solutions.ipynb` | ⚠️ Minimal | 1 |
| 5 | `Chapter05_Solutions.ipynb` | ✅ Complete | 21 |
| 6 | `Chapter06_Solutions.ipynb` | ✅ Complete | 23 |
| 8 | `Chapter08_Solutions.ipynb` | ✅ Complete | 13 |
| 9 | `Chapter09_Solutions.ipynb` | ✅ Complete | 10 |
| 11 | `Chapter11_Solutions.ipynb` | ✅ Complete | 13 |

**Missing chapters:**
- Chapter 2 (Software) - Installation, no exercises
- Chapter 7 (Basic Algorithms) - WIP
- Chapter 10 (Shor's) - WIP
- Chapter 12 (Quantum Tests) - WIP
- Chapter 13 (VQLS) - WIP

---

## 🎯 Pedagogical Notes

### **Solutions Support Critical Thinking**

This textbook emphasizes **critical evaluation** of quantum algorithms. Solutions should reinforce this:

**Example from Chapter 12 (when complete):**
```python
# Exercise: Estimate measurement overhead for VQLS

# Solution should include:
# 1. Calculate O(1/ε²) for different precision levels
# 2. Compare with classical conjugate gradient O(κ log(1/ε))
# 3. Discuss when quantum overhead dominates
# 4. Critical conclusion about practical feasibility
```

**Don't just provide code** - provide **reasoning and analysis**.

### **Solutions Should Show Best Practices**

- Use utilities (don't redefine functions)
- Include comments explaining the approach
- Add assertions/tests to verify correctness
- Discuss limitations and edge cases
- Compare quantum vs. classical when relevant

---

## 🔄 Updating Solutions

### **Adding New Solutions**

1. **Create solution in appropriate chapter notebook**
   ```bash
   jupyter notebook solutions/Chapter07_Solutions.ipynb
   ```

2. **Follow the pattern:**
   - Exercise description (markdown)
   - Solution code (with comments)
   - Output/results
   - Discussion/analysis (if appropriate)

3. **Use utilities** whenever possible:
   ```python
   # Good: Reuse existing functions
   counts = simulateCircuit(circuit, shots=1000)
   
   # Bad: Redefine the same function
   def my_simulate_circuit(...):  # Don't do this!
   ```

### **Maintaining Solutions Across Textbook Editions**

- Solutions are **separate from main notebooks** (good for versioning)
- Update solutions when:
  - Main chapter changes significantly
  - Utilities API changes
  - New exercises added to textbook
  - Better solution approaches discovered

---

## ⚠️ Important Notes

### **For Instructors**

1. **Academic Integrity:** Decide your policy on solution access
2. **Customization:** Feel free to modify for your course needs
3. **Contributions:** Add your own solutions for missing chapters
4. **Feedback:** Share better solution approaches with the author

### **For Students**

1. **Honor Code:** Use solutions ethically (after attempting problems)
2. **Understanding:** Focus on learning, not just getting answers
3. **Practice:** Solutions are learning tools, not shortcuts

---

## 📧 Contributing Solutions

Have a better solution? Found an error? Want to contribute solutions for missing chapters?

1. Open an issue describing the improvement
2. Submit a pull request with your solution
3. Follow the existing format and style
4. Include explanation/analysis with code

---

## 📜 License

Solutions are provided for educational use in conjunction with the textbook.

Instructors may modify and redistribute solutions for their courses.

---

**Remember:** Solutions are most valuable when used as a **learning tool**, not a shortcut!

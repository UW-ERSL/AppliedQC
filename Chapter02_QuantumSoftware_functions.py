"""
Chapter 2 — Quantum Software: companion functions.

Provides the classical oracle used in the Bernstein–Vazirani demonstration
of Chapter 2. The oracle hides a fixed secret bit string ``s`` and returns
the mod-2 inner product ``s · x``, the exact function the Bernstein–Vazirani
quantum algorithm recovers in a single query.
"""


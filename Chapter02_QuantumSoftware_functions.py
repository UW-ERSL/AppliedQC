"""
Chapter 2 — Quantum Software: companion functions.

Provides the classical oracle used in the Bernstein–Vazirani demonstration
of Chapter 2. The oracle hides a fixed secret bit string ``s`` and returns
the mod-2 inner product ``s · x``, the exact function the Bernstein–Vazirani
quantum algorithm recovers in a single query.
"""


## Berstein-Vazirani function

def secretBitStringFunction(x):
    """
    Evaluate the Bernstein–Vazirani oracle f(x) = s · x (mod 2).

    Computes the mod-2 inner product of the input bit string ``x`` with the
    hidden secret string ``s = '11010'``. This is the classical function that
    the Bernstein–Vazirani algorithm learns with a single quantum query,
    whereas classically it would require one query per bit.

    Parameters
    ----------
    x : sequence of int
        Input bits, one per position (length 5 to match ``s``). Each element
        is treated as an integer 0 or 1.

    Returns
    -------
    int
        The parity ``(Σ_i s_i · x_i) mod 2`` (0 or 1). If ``len(x)`` does not
        match ``len(s)``, prints an error message and returns 0.
    """
    s = '11010'
    if (len(x) != len(s)):
        print('Error: Length of secret string is', len(s))
        return 0
    a = 0
    for i in range(len(s)):
        a = a + int(s[i])*x[i]
    return a%2
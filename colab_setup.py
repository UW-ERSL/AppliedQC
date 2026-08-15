"""
Colab environment setup for "Applied Quantum Computing for Engineers".

Not run directly. Each notebook's first cell clones the repository and
then exec()s this file, so this is the single place where the setup
logic lives -- change it here and every notebook picks it up, because
every Colab session clones the repository fresh.

Deliberately free of IPython magics (%pip, !git) so it can be exec'd
from any context.
"""
import os
import re
import subprocess
import sys

REPO          = "/content/AppliedQC"
STAMP_INSTALL = "/content/.appliedqc_installed"
STAMP_RESTART = "/content/.appliedqc_restarted"

BAR = "  " + "=" * 58


def _install():
    if os.path.exists(STAMP_INSTALL):
        return
    print("Installing the packages this book needs. Takes a minute or two.")
    print("A red block about 'numba' may appear. It is harmless: that is")
    print("Colab's own numba, which this book never uses.\n")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r",
         os.path.join(REPO, "requirements.txt")],
        check=True,
    )
    open(STAMP_INSTALL, "w").close()


def _wanted_numpy():
    with open(os.path.join(REPO, "requirements.txt")) as fh:
        return re.search(r"^numpy==(\S+)", fh.read(), re.M).group(1)


def _restart_session():
    """Restart the kernel so the newly installed NumPy becomes active."""
    print("\n" + BAR)
    print("   Setup complete. Restarting the session now -- this is normal.")
    print("")
    print("   When the restart finishes (a few seconds), run this cell")
    print("   again. It will be quick, and the notebook will then run")
    print("   from top to bottom without interruption.")
    print(BAR + "\n")
    sys.stdout.flush()

    open(STAMP_RESTART, "w").close()
    import IPython
    IPython.Application.instance().kernel.do_shutdown(True)


_install()

sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy                                                   # noqa: E402

# Colab imports its own NumPy before the first cell runs, so a freshly
# installed NumPy is not active until the kernel restarts. Left unchecked,
# the notebook fails much later and deep inside SciPy, with an ImportError
# about 'numpy._core.umath' that gives no hint of the real cause.
want = _wanted_numpy()

if numpy.__version__ == want:
    import scipy, qiskit                                       # noqa: E402
    print("Ready.  numpy", numpy.__version__,
          "| scipy", scipy.__version__,
          "| qiskit", qiskit.__version__)

elif not os.path.exists(STAMP_RESTART):
    # Nothing after this runs: the kernel is on its way down.
    _restart_session()

else:
    # Already restarted once and NumPy is still wrong: something is
    # genuinely off, so say so rather than restarting forever.
    raise RuntimeError(
        f"\n\n{BAR}\n"
        f"   NumPy {numpy.__version__} is active, but this book needs {want},\n"
        "   and restarting the session did not fix it.\n"
        "\n"
        "   Try:  Runtime > Disconnect and delete runtime,\n"
        "   then open the notebook again and re-run this cell.\n"
        f"{BAR}\n"
    )
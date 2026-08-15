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

REPO  = "/content/AppliedQC"
STAMP = "/content/.appliedqc_installed"


class RestartSessionRequired(RuntimeError):
    """The kernel still holds Colab's NumPy, not the book's."""


def _install():
    if os.path.exists(STAMP):
        print("Packages already installed in this session -- skipping pip.")
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r",
         os.path.join(REPO, "requirements.txt")],
        check=True,
    )
    open(STAMP, "w").close()


def _check_restart(numpy_version):
    # Colab imports its own NumPy before the first cell runs, so a freshly
    # installed NumPy is not live until the kernel restarts. Without this
    # check the notebook fails much later, deep inside SciPy, with an
    # ImportError about 'numpy._core.umath' that gives no hint of the cause.
    with open(os.path.join(REPO, "requirements.txt")) as fh:
        want = re.search(r"^numpy==(\S+)", fh.read(), re.M).group(1)

    if numpy_version != want:
        raise RestartSessionRequired(
            "\n\n"
            "  ============================================================\n"
            "   Setup finished. One more step -- this is expected.\n"
            "\n"
            f"   Colab loaded NumPy {numpy_version} before the install ran,\n"
            f"   so the book's NumPy {want} is on disk but not yet in use.\n"
            "\n"
            "     ->  Runtime  >  Restart session\n"
            "     ->  then run this cell again\n"
            "\n"
            "   The rest of the notebook will work after that.\n"
            "  ============================================================\n"
        )


_install()

sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy, scipy, qiskit                                    # noqa: E402
print("numpy", numpy.__version__, "| scipy", scipy.__version__,
      "| qiskit", qiskit.__version__)

_check_restart(numpy.__version__)
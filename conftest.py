"""Pytest bootstrap: ensure the repo root is importable as the package root.

Lets `import data.load`, `import models...` resolve when running `pytest` from a
clean clone regardless of the invoking working directory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

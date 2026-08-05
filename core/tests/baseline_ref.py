"""Read-only loader for LaGoVAD baseline modules (parity-test harness).

Registers a synthetic package pointing at ``LaGoVAD-PreVAD/src/models/LaGoVAD``
so its submodules import with working relative imports *without* executing the
package ``__init__.py`` (which requires lightning). The baseline tree is never
modified — modules are only instantiated in-memory for parity checks.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

BASELINE_LAGOVAD_DIR = (
    Path(__file__).resolve().parents[2] / "LaGoVAD-PreVAD" / "src" / "models" / "LaGoVAD"
)
_PKG_NAME = "lagovad_ref"


def load_baseline_module(module_name: str) -> Any:
    """Import ``LaGoVAD/<module_name>.py`` under the synthetic ``lagovad_ref`` package."""
    if _PKG_NAME not in sys.modules:
        package = types.ModuleType(_PKG_NAME)
        package.__path__ = [str(BASELINE_LAGOVAD_DIR)]
        sys.modules[_PKG_NAME] = package
    return importlib.import_module(f"{_PKG_NAME}.{module_name}")


__all__ = ["BASELINE_LAGOVAD_DIR", "load_baseline_module"]

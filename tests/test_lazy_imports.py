from __future__ import annotations

import importlib
import subprocess
import sys


def _clear_raganything_modules() -> None:
    for name in list(sys.modules):
        if name == "raganything" or name.startswith("raganything."):
            sys.modules.pop(name, None)


def test_import_genealogy_does_not_force_heavy_modules() -> None:
    had_lightrag = "lightrag" in sys.modules
    had_mineru = "mineru" in sys.modules

    _clear_raganything_modules()

    importlib.import_module("raganything.genealogy")

    if not had_lightrag:
        assert "lightrag" not in sys.modules
    if not had_mineru:
        assert "mineru" not in sys.modules


def test_import_raganything_and_genealogy_in_subprocess() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import raganything; import raganything.genealogy"],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr

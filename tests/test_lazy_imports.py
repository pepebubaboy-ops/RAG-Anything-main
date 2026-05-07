from __future__ import annotations

import importlib
import subprocess
import sys


def _clear_raganything_modules() -> None:
    for name in list(sys.modules):
        if name == "raganything" or name.startswith("raganything."):
            sys.modules.pop(name, None)


def test_import_genealogy_does_not_force_optional_modules() -> None:
    optional_modules = ("mineru", "openai", "neo4j")
    was_loaded = {name: name in sys.modules for name in optional_modules}

    _clear_raganything_modules()

    importlib.import_module("raganything.genealogy")

    for name, already_loaded in was_loaded.items():
        if not already_loaded:
            assert name not in sys.modules


def test_import_raganything_and_genealogy_in_subprocess() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import raganything; import raganything.genealogy"],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr

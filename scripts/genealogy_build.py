from __future__ import annotations

import sys
import warnings

from raganything.cli import main


if __name__ == "__main__":
    warnings.warn(
        "scripts/genealogy_build.py is deprecated. Use `genealogy-rag genealogy build ...`.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise SystemExit(main(["genealogy", "build", *sys.argv[1:]]))

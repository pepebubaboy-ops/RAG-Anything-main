from __future__ import annotations

import warnings

from raganything.cli import main


if __name__ == "__main__":
    warnings.warn(
        "scripts/raganything_cli.py is a thin wrapper. Prefer the `raganything` CLI entrypoint.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise SystemExit(main())

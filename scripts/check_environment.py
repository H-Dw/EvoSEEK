#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import platform
import sys


def main() -> None:
    packages = {}
    for name in ("numpy", "pandas", "sklearn", "scipy", "yaml", "httpx"):
        module = importlib.import_module(name)
        packages[name] = getattr(module, "__version__", "available")
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "packages": packages,
                "ok": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


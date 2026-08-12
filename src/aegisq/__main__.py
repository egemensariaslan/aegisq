"""Entry point for ``python -m aegisq``.

The ``__name__`` guard matters here: test collectors and doc tooling import every
module in the package, and an unguarded call would launch the CLI mid-import.
Running via ``-m`` sets ``__name__`` to ``"__main__"``, so the guard still fires.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

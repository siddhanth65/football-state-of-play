"""Full Week-2 dataset build: possession index -> labelled possessions.

Single entry point that runs :func:`data.possessions.main` then
:func:`data.labels.main` in one process, so the cached
``possessions_index.parquet`` from the first stage feeds the second. Used for the
detached background build (and equivalent to the two-step ``make data`` target).

Run as ``python -m data.build``.
"""

from __future__ import annotations

from data import labels, possessions


def main() -> None:
    """Build the possession index, then attach labels + the xT grid."""
    possessions.main()
    labels.main()


if __name__ == "__main__":
    main()

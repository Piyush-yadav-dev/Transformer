"""Support the same CLI through python -m transformer."""

from transformer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

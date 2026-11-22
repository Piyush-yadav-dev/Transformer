"""Start the local website using this checkout's data and model registry."""

import os
from pathlib import Path

from transformer.web.__main__ import main

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()

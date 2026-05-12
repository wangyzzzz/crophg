import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
SRC = ROOT / "src"
for path in [str(SRC), str(ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from models.result34.parallel_launcher import main


if __name__ == "__main__":
    main()

"""Document Processing Platform Backend Package."""
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
repo_root = backend_dir.parent
for p in (str(backend_dir), str(repo_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

__version__ = "0.1.0"

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
smoke_val = PROJECT_ROOT / "training" / "datasets" / "smoke_kannada_val.jsonl"

samples = [json.loads(line) for line in smoke_val.read_text(encoding="utf-8").splitlines() if line.strip()]
print(f"Total available smoke val samples: {len(samples)}")
for i, s in enumerate(samples[:25]):
    img = s.get("image", "")
    txt = s.get("text", "")
    print(f"[{i:02d}] {txt} (len={len(txt)}) -> {img}")

"""Standalone Dataset Readiness Audit Script.

Usage:
    python evaluation/scripts/audit_datasets.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.dataset_audit import run_dataset_audit


def main():
    print("=" * 70)
    print("  LAND RECORD DIGITIZATION - DATASET READINESS AUDIT")
    print("=" * 70)

    report = run_dataset_audit(root_dir=PROJECT_ROOT)
    print(report.to_markdown())
    print("=" * 70)


if __name__ == "__main__":
    main()

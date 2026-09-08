"""Lightweight test runner that executes unit tests and reports exact progress to stdout."""

import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_tests(patterns=None):
    loader = unittest.TestLoader()
    start_dir = PROJECT_ROOT / "tests" / "unit"

    if patterns:
        suite = unittest.TestSuite()
        for pat in patterns:
            if pat.endswith(".py"):
                mod_path = Path(pat)
                if not mod_path.is_absolute():
                    mod_path = PROJECT_ROOT / pat
                # Import module
                rel_parts = mod_path.relative_to(PROJECT_ROOT).with_suffix("").parts
                mod_name = ".".join(rel_parts)
                mod_suite = loader.loadTestsFromName(mod_name)
                suite.addTests(mod_suite)
            else:
                s = loader.discover(str(start_dir), pattern=pat)
                suite.addTests(s)
    else:
        suite = loader.discover(str(start_dir), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 70, flush=True)
    print(f"Total Tests Run : {result.testsRun}", flush=True)
    print(f"Passed          : {result.testsRun - len(result.errors) - len(result.failures)}", flush=True)
    print(f"Failures        : {len(result.failures)}", flush=True)
    print(f"Errors          : {len(result.errors)}", flush=True)
    print("=" * 70, flush=True)

    return result.wasSuccessful()


if __name__ == "__main__":
    test_args = sys.argv[1:] if len(sys.argv) > 1 else None
    success = run_tests(test_args)
    if not success:
        sys.exit(1)

"""One-shot local check for icu-decision-agent.

Runs the harness preflight then the no-DB test subset. Exit 0 only if both pass.
Usage: .venv\\Scripts\\python.exe scripts/check_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = Path.home() / ".dsh" / "skills" / "complex-project-harness" / "scripts" / "preflight.py"
TESTS = [
    "tests/test_lare.py",
    "tests/test_lar_policy.py",
    "tests/test_lare_runner.py",
    "tests/test_predict.py",
    "tests/test_features_leak.py",
    "tests/test_dca.py",
]


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def main() -> int:
    code = run([sys.executable, str(PREFLIGHT), "--config", str(ROOT / "preflight.yaml")])
    if code:
        print("PREFLIGHT FAILED")
        return code
    return run([sys.executable, "-m", "pytest", *TESTS, "-q"])


if __name__ == "__main__":
    raise SystemExit(main())

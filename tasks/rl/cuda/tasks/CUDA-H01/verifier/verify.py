#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
import importlib.util

TASK_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_DIR = TASK_ROOT / "verifier"
VERIFIER_IMPL = VERIFIER_DIR / "verifier.py"
ORACLE_RESULT = TASK_ROOT / "oracle_result.json"

if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

spec = importlib.util.spec_from_file_location(
    "cuda_h01_verifier_impl",
    VERIFIER_IMPL,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load verifier implementation: {VERIFIER_IMPL}")

module = importlib.util.module_from_spec(spec)

# IMPORTANT:
# Register the module before exec_module().
# dataclasses on Python 3.13 may inspect sys.modules while processing
# annotations/classes.
sys.modules[spec.name] = module
spec.loader.exec_module(module)

Verifier = getattr(module, "Verifier", None)
if Verifier is None:
    raise TypeError("verifier/verifier.py does not expose Verifier")


def load_oracle_result() -> dict:
    if not ORACLE_RESULT.exists():
        raise FileNotFoundError(
            f"Oracle result not found: {ORACLE_RESULT}"
        )

    data = json.loads(ORACLE_RESULT.read_text())
    if not isinstance(data, dict):
        raise TypeError("oracle_result.json must contain a JSON object")
    return data


def main() -> int:
    result = load_oracle_result()

    trajectory = result.get("trajectory", [])
    final_state = result.get("final_state", {})

    verifier = Verifier()
    verdict = verifier.verify(trajectory, final_state)

    if hasattr(verdict, "passed"):
        passed = bool(verdict.passed)
        payload = {
            "verified": passed,
        }
        if hasattr(verdict, "score"):
            payload["score"] = verdict.score
        if hasattr(verdict, "reasons"):
            payload["reasons"] = list(verdict.reasons)
    elif isinstance(verdict, bool):
        passed = verdict
        payload = {"verified": passed}
    elif isinstance(verdict, dict):
        payload = verdict
        passed = bool(
            verdict.get("verified", verdict.get("passed", False))
        )
    else:
        raise TypeError(
            f"Unsupported verifier result type: {type(verdict).__name__}"
        )

    print(json.dumps(payload, indent=2))

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

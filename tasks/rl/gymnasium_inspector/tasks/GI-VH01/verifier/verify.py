#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_DIR = Path(__file__).resolve().parent
VERIFIER_PATH = VERIFIER_DIR / "verifier.py"
ORACLE_RESULT_PATH = TASK_ROOT / "oracle_result.json"

sys.path.insert(0, str(TASK_ROOT))

MODULE_NAME = "gi_vh01_verifier_impl"

spec = importlib.util.spec_from_file_location(
    MODULE_NAME,
    VERIFIER_PATH,
)

if spec is None or spec.loader is None:
    raise RuntimeError(
        f"Unable to create import specification for {VERIFIER_PATH}"
    )

module = importlib.util.module_from_spec(spec)

# Required for Python 3.13 dataclass/module handling.
sys.modules[MODULE_NAME] = module

spec.loader.exec_module(module)


def load_oracle_result() -> dict:
    if not ORACLE_RESULT_PATH.exists():
        raise FileNotFoundError(
            f"Oracle result not found: {ORACLE_RESULT_PATH}"
        )

    with ORACLE_RESULT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError("oracle_result.json must contain a JSON object")

    return data


def main() -> int:
    verifier_cls = getattr(module, "Verifier", None)

    if verifier_cls is None:
        raise RuntimeError(
            "verifier.py does not expose a Verifier class"
        )

    verifier = verifier_cls()

    verify_method = getattr(verifier, "verify", None)

    if verify_method is None:
        raise RuntimeError(
            "Verifier class does not expose a verify() method"
        )

    oracle_result = load_oracle_result()

    trajectory = oracle_result.get("trajectory", [])
    final_state = oracle_result.get("final_state")

    if not isinstance(trajectory, list):
        raise TypeError("Oracle trajectory must be a list")

    if final_state is None:
        raise ValueError("Oracle result is missing final_state")

    result = verify_method(
        trajectory,
        final_state,
    )

    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True))
        verified = bool(result.get("verified", False))
    else:
        verified = bool(result)
        print(
            json.dumps(
                {"verified": verified},
                indent=2,
            )
        )

    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())

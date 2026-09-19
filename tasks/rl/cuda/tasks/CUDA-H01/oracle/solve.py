#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
import importlib.util

TASK_ROOT = Path(__file__).resolve().parents[1]
ORACLE_DIR = TASK_ROOT / "oracle"
ORACLE_IMPL = ORACLE_DIR / "oracle.py"

if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

spec = importlib.util.spec_from_file_location(
    "cuda_h01_oracle_impl",
    ORACLE_IMPL,
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load oracle implementation: {ORACLE_IMPL}")

module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

solve = getattr(module, "solve", None)
if not callable(solve):
    raise TypeError("oracle/oracle.py does not expose callable solve(seed)")


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    result = solve(seed)

    if not isinstance(result, dict):
        raise TypeError("Oracle solve() must return a dict")

    print(json.dumps(result, indent=2, sort_keys=False))

    result_path = TASK_ROOT / "oracle_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

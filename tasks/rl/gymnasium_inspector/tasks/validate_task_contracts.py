from __future__ import annotations

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parent
EXPECTED_DIFFICULTIES = {
    "M": 11,
    "H": 11,
    "VH": 11,
}


REQUIRED_TOP_LEVEL = {
    "task_id",
    "name",
    "domain",
    "difficulty",
    "environment",
    "objective",
    "agent_interface",
    "oracle",
    "verifier",
    "reward",
    "evaluation",
}


REQUIRED_ORACLE = {
    "enabled",
    "reference_behavior",
    "deterministic",
    "independent_from_agent",
}


REQUIRED_VERIFIER = {
    "enabled",
    "checks",
    "oracle_comparison",
    "isolated",
}


REQUIRED_REWARD = {
    "type",
    "components",
    "aggregation",
    "terminal",
}


REQUIRED_EVALUATION = {
    "split",
    "success_criteria",
    "max_steps",
}


def fail(message: str) -> None:
    print(f"ERROR: {message}")
    raise SystemExit(1)


def validate_task(path: Path) -> str:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"{path.name}: invalid YAML: {exc}")

    if not isinstance(data, dict):
        fail(f"{path.name}: task must be a YAML mapping")

    missing = REQUIRED_TOP_LEVEL - set(data)
    if missing:
        fail(f"{path.name}: missing top-level fields: {sorted(missing)}")

    task_id = data["task_id"]
    if not isinstance(task_id, str) or not task_id:
        fail(f"{path.name}: task_id must be a non-empty string")

    difficulty = data["difficulty"]
    if difficulty not in EXPECTED_DIFFICULTIES:
        fail(
            f"{path.name}: difficulty must be one of "
            f"{sorted(EXPECTED_DIFFICULTIES)}"
        )

    if data["domain"] != "gymnasium_inspector":
        fail(
            f"{path.name}: domain must be "
            f"'gymnasium_inspector'"
        )

    environment = data["environment"]
    if not isinstance(environment, dict):
        fail(f"{path.name}: environment must be a mapping")

    for field in ("type", "id", "version"):
        if field not in environment:
            fail(
                f"{path.name}: environment missing '{field}'"
            )

    oracle = data["oracle"]
    if not isinstance(oracle, dict):
        fail(f"{path.name}: oracle must be a mapping")

    missing = REQUIRED_ORACLE - set(oracle)
    if missing:
        fail(
            f"{path.name}: oracle missing fields: "
            f"{sorted(missing)}"
        )

    if oracle["enabled"] is not True:
        fail(f"{path.name}: oracle.enabled must be true")

    if oracle["deterministic"] is not True:
        fail(
            f"{path.name}: oracle.deterministic must be true"
        )

    if oracle["independent_from_agent"] is not True:
        fail(
            f"{path.name}: oracle.independent_from_agent "
            f"must be true"
        )

    verifier = data["verifier"]
    if not isinstance(verifier, dict):
        fail(f"{path.name}: verifier must be a mapping")

    missing = REQUIRED_VERIFIER - set(verifier)
    if missing:
        fail(
            f"{path.name}: verifier missing fields: "
            f"{sorted(missing)}"
        )

    if verifier["enabled"] is not True:
        fail(f"{path.name}: verifier.enabled must be true")

    if verifier["isolated"] is not True:
        fail(f"{path.name}: verifier.isolated must be true")

    checks = verifier["checks"]
    if not isinstance(checks, list) or not checks:
        fail(
            f"{path.name}: verifier.checks must be "
            f"a non-empty list"
        )

    reward = data["reward"]
    if not isinstance(reward, dict):
        fail(f"{path.name}: reward must be a mapping")

    missing = REQUIRED_REWARD - set(reward)
    if missing:
        fail(
            f"{path.name}: reward missing fields: "
            f"{sorted(missing)}"
        )

    components = reward["components"]
    if not isinstance(components, list) or not components:
        fail(
            f"{path.name}: reward.components must be "
            f"a non-empty list"
        )

    if reward["aggregation"] != "sum":
        fail(
            f"{path.name}: reward.aggregation must be 'sum'"
        )

    evaluation = data["evaluation"]
    if not isinstance(evaluation, dict):
        fail(f"{path.name}: evaluation must be a mapping")

    missing = REQUIRED_EVALUATION - set(evaluation)
    if missing:
        fail(
            f"{path.name}: evaluation missing fields: "
            f"{sorted(missing)}"
        )

    if evaluation["split"] not in {
        "train",
        "validation",
        "held_out",
    }:
        fail(
            f"{path.name}: evaluation.split must be "
            f"train, validation, or held_out"
        )

    if not isinstance(evaluation["max_steps"], int):
        fail(
            f"{path.name}: evaluation.max_steps must be int"
        )

    if evaluation["max_steps"] <= 0:
        fail(
            f"{path.name}: evaluation.max_steps must be > 0"
        )

    return difficulty


def main() -> None:
    files = sorted(ROOT.glob("GI-*.yaml"))

    if len(files) != 33:
        fail(
            f"expected exactly 33 task files, found {len(files)}"
        )

    counts = {
        difficulty: 0
        for difficulty in EXPECTED_DIFFICULTIES
    }

    ids: set[str] = set()

    for path in files:
        difficulty = validate_task(path)

        data = yaml.safe_load(
            path.read_text(encoding="utf-8")
        )

        task_id = data["task_id"]

        if task_id in ids:
            fail(f"duplicate task_id: {task_id}")

        ids.add(task_id)
        counts[difficulty] += 1

    if counts != EXPECTED_DIFFICULTIES:
        fail(
            "difficulty distribution mismatch: "
            f"{counts} != {EXPECTED_DIFFICULTIES}"
        )

    expected_ids = (
        {f"GI-M{i:02d}" for i in range(1, 12)}
        | {f"GI-H{i:02d}" for i in range(1, 12)}
        | {f"GI-VH{i:02d}" for i in range(1, 12)}
    )

    if ids != expected_ids:
        missing = sorted(expected_ids - ids)
        extra = sorted(ids - expected_ids)

        if missing:
            print(f"Missing task IDs: {missing}")

        if extra:
            print(f"Unexpected task IDs: {extra}")

        fail("task ID set does not match the frozen 33-task matrix")

    print("=== GYMNASIUM / INSPECTOR TASK CONTRACT VALIDATION ===")
    print(f"Task files: {len(files)}")
    print(f"Medium: {counts['M']}")
    print(f"Hard: {counts['H']}")
    print(f"Very Hard: {counts['VH']}")
    print("Oracle: ENABLED / DETERMINISTIC / INDEPENDENT")
    print("Verifier: ENABLED / ISOLATED")
    print("Reward: MULTI-COMPONENT / SUM AGGREGATION")
    print("=== TASK CONTRACTS STRUCTURALLY VALID ===")
    print("=== TERMINAL REMAINS OPEN ===")


if __name__ == "__main__":
    main()

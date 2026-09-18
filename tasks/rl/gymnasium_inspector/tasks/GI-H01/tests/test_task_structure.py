from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_required_files_exist():
    required = [
        "task.yaml",
        "instruction.md",
        "environment/data/gym_env.py",
        "oracle/solve.py",
        "verifier/verify.py",
        "tests/test_gym_env.py",
        "tests/test_outputs.py",
        "tests/test_multi_reward.py",
        "tests/test_hidden_boundary.py",
        "tests/test_trajectory_audit.py",
        "tests/test_anti_cheat.py",
        "tests/test_oracle.py",
    ]

    for relative in required:
        assert (ROOT / relative).exists(), relative


def test_task_yaml_contract():
    payload = yaml.safe_load(
        (ROOT / "task.yaml").read_text(
            encoding="utf-8"
        )
    )

    required = {
        "task_id",
        "version",
        "domain",
        "difficulty",
        "status",
        "name",
        "objective",
        "agent_interface",
        "environment",
        "evaluation",
        "oracle",
        "reward",
        "verifier",
        "anti_cheat",
    }

    assert required.issubset(payload)
    assert payload["task_id"] == "GI-H01"
    assert payload["difficulty"] == "hard"
    assert payload["reward"]["type"] == "multi_reward"
    assert payload["oracle"]["deterministic"] is True
    assert payload["verifier"]["hidden"] is True
    assert payload["anti_cheat"]["enabled"] is True

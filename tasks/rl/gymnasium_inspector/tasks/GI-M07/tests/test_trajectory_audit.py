from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_trajectory_audit_fields():
    env = GymInspectorEnv(max_steps=8)

    observation, info = env.reset(seed=123)

    records = []

    for action in [1, 1, 1]:
        next_observation, reward, terminated, truncated, step_info = env.step(action)

        records.append(
            {
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": step_info,
            }
        )

        observation = next_observation

    assert len(records) == 3

    for record in records:
        assert isinstance(record["reward"], float)
        assert isinstance(record["terminated"], bool)
        assert isinstance(record["truncated"], bool)
        assert isinstance(record["info"], dict)

    env.close()

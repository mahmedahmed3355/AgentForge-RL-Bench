import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_trajectory_records_are_auditable():
    env = GymInspectorEnv(max_steps=7)

    env.reset(seed=123)

    records = []

    for action in [1, 1, 1]:
        observation, reward, terminated, truncated, info = env.step(action)

        records.append(
            {
                "position": int(observation["position"][0]),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )

    assert len(records) == 3

    for record in records:
        assert isinstance(record["reward"], float)
        assert isinstance(record["terminated"], bool)
        assert isinstance(record["truncated"], bool)
        assert isinstance(record["info"], dict)

    assert records[-1]["position"] == 3

    env.close()

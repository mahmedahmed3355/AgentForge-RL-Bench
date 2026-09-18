from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_trajectory_records_contract_values():
    env = GymInspectorEnv(max_steps=6)
    env.reset(seed=123)

    records = []

    for action in [1, 1, 1, 1, 1, 1]:
        observation, reward, terminated, truncated, info = env.step(action)

        records.append(
            {
                "observation": observation.copy(),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": dict(info),
            }
        )

        if terminated or truncated:
            break

    assert len(records) == 6
    assert records[-1]["terminated"] is True
    assert records[-1]["truncated"] is False

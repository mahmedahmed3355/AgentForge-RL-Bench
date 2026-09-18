from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_trajectory_audit():
    env = GymInspectorEnv(max_steps=7)

    env.reset(seed=123)

    trajectory = []

    for action in [1, 1, 1, 1, 1, 1, 1]:
        observation, reward, terminated, truncated, info = env.step(action)

        trajectory.append(
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

    assert len(trajectory) == 7
    assert trajectory[-1]["terminated"] is True
    assert trajectory[-1]["truncated"] is False

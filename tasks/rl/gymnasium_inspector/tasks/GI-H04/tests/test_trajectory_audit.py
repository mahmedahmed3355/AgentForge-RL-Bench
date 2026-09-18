import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import DependencyGraphEnv


def test_valid_topological_trajectory():
    env = DependencyGraphEnv(max_steps=8)
    env.reset(seed=123)

    trajectory = []

    for action in [0, 1, 2, 3, 4, 5]:
        observation, reward, terminated, truncated, info = env.step(action)

        trajectory.append(
            (
                action,
                observation["progress"],
                terminated,
                truncated,
                info["valid_action"],
            )
        )

    assert [x[1] for x in trajectory] == [1, 2, 3, 4, 5, 6]
    assert all(x[4] for x in trajectory)
    assert trajectory[-1][2] is True
    assert trajectory[-1][3] is False

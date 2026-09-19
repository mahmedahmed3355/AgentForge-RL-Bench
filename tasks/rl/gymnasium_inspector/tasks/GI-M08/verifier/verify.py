from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def verify() -> bool:
    env = GymInspectorEnv(max_steps=7)

    obs1, info1 = env.reset(seed=123)
    trajectory_1 = []

    for action in [1, 1, 1, 1, 1, 1, 1]:
        item = env.step(action)
        trajectory_1.append(item)

        if item[2] or item[3]:
            break

    env.reset(seed=123)
    trajectory_2 = []

    for action in [1, 1, 1, 1, 1, 1, 1]:
        item = env.step(action)
        trajectory_2.append(item)

        if item[2] or item[3]:
            break

    assert len(trajectory_1) == len(trajectory_2)

    for a, b in zip(trajectory_1, trajectory_2):
        assert a[0]["position"].tolist() == b[0]["position"].tolist()
        assert a[0]["remaining"] == b[0]["remaining"]
        assert a[1] == b[1]
        assert a[2] == b[2]
        assert a[3] == b[3]

        observation, reward, terminated, truncated, info = a

        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert not (terminated and truncated)

    assert trajectory_1[-1][2] is True
    assert trajectory_1[-1][3] is False

    env.close()

    return True


if __name__ == "__main__":
    print({"verified": verify()})

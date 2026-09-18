from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def verify() -> bool:
    env = GymInspectorEnv(max_steps=8)

    observation_a, info_a = env.reset(seed=123)
    observation_b, info_b = env.reset(seed=123)

    assert env.observation_space.contains(observation_a)
    assert env.observation_space.contains(observation_b)
    assert observation_a["state"].tolist() == observation_b["state"].tolist()

    terminated = False
    truncated = False

    for action in [1, 1, 1, 1, 1, 1, 1, 1]:
        observation, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert not (terminated and truncated)

        if terminated or truncated:
            break

    assert terminated is True
    assert truncated is False

    env.close()
    return True


if __name__ == "__main__":
    print({"verified": verify()})

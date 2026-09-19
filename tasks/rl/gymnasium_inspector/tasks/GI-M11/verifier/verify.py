from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def main() -> int:
    env = GymInspectorEnv(max_steps=5)

    observation, info = env.reset(seed=123)

    assert isinstance(info, dict)
    assert env.observation_space.contains(observation)

    terminated = False
    truncated = False

    for action in [1, 1, 1, 1, 1]:
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

    print('{"verified": true}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

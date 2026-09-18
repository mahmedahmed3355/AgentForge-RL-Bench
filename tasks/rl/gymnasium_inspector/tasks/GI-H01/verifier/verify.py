from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def main() -> int:
    env = GymInspectorEnv(max_steps=6)

    observation, info = env.reset(seed=123)

    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)

    trajectory = []

    for action in [1, 1, 1, 1, 1, 1]:
        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert not (terminated and truncated)

        trajectory.append(
            (
                int(info["step"]),
                int(info["position"]),
                bool(terminated),
                bool(truncated),
            )
        )

        if terminated or truncated:
            break

    assert trajectory[-1][0] == 6
    assert trajectory[-1][1] == 6
    assert trajectory[-1][2] is True
    assert trajectory[-1][3] is False

    env.close()

    print('{"verified": true}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

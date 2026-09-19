from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def main() -> int:
    actions = [1, 1, 1, 1, 1, 1]

    env = GymInspectorEnv(max_steps=8)

    obs_a, info_a = env.reset(seed=123)

    assert env.observation_space.contains(obs_a)
    assert isinstance(info_a, dict)

    trajectory_a = []

    for action in actions:
        assert env.action_space.contains(action)

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert not (terminated and truncated)

        trajectory_a.append(
            (
                int(obs["state"][0]),
                int(obs["step"][0]),
                float(reward),
                bool(terminated),
                bool(truncated),
            )
        )

        if terminated or truncated:
            break

    env.close()

    env = GymInspectorEnv(max_steps=8)

    obs_b, info_b = env.reset(seed=123)

    trajectory_b = []

    for action in actions:
        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        trajectory_b.append(
            (
                int(obs["state"][0]),
                int(obs["step"][0]),
                float(reward),
                bool(terminated),
                bool(truncated),
            )
        )

        if terminated or truncated:
            break

    assert int(obs_a["state"][0]) == int(obs_b["state"][0])
    assert int(obs_a["step"][0]) == int(obs_b["step"][0])
    assert info_a == info_b
    assert trajectory_a == trajectory_b

    env.reset(seed=999)

    for _ in range(8):
        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(0)

        assert not (terminated and truncated)

        if terminated or truncated:
            break

    assert terminated is False
    assert truncated is True

    env.close()

    print(json.dumps({"verified": True}))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

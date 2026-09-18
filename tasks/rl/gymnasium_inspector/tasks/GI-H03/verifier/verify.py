from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import BranchingIrreversibleEnv


def verify() -> dict:
    env = BranchingIrreversibleEnv(max_steps=10)

    observation, info = env.reset(seed=123)

    assert isinstance(info, dict)
    assert env.observation_space.contains(observation)

    actions = [0, 1, 2, 3]

    terminated = False
    truncated = False
    total_reward = 0.0

    for action in actions:
        assert env.action_space.contains(action)

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

        total_reward += reward

        if terminated or truncated:
            break

    assert terminated is True
    assert truncated is False
    assert int(observation["state"]) == 5
    assert int(observation["resources"][0]) == 3
    assert total_reward == 9.5

    env.close()

    return {"verified": True}


def main() -> int:
    result = verify()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

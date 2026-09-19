import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "data"))

from gym_env import GymInspectorEnv


def test_multi_reward_signal():
    env = GymInspectorEnv()

    env.reset(seed=123)

    rewards = []

    for action in [1, 1, 1, 1, 1, 1]:
        _, reward, terminated, truncated, _ = env.step(action)

        rewards.append(float(reward))

        if terminated or truncated:
            break

    assert rewards
    assert all(isinstance(value, float) for value in rewards)
    assert sum(rewards) > 0.0

    env.close()

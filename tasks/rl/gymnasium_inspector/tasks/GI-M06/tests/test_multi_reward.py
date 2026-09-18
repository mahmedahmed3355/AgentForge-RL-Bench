from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_progress_and_terminal_reward_components():
    env = GymInspectorEnv(max_steps=7)

    env.reset(seed=123)

    rewards = []

    for _ in range(7):
        _, reward, terminated, truncated, _ = env.step(1)

        rewards.append(reward)

        if terminated or truncated:
            break

    assert len(rewards) == 7
    assert all(isinstance(value, float) for value in rewards)
    assert rewards[-1] > rewards[0]

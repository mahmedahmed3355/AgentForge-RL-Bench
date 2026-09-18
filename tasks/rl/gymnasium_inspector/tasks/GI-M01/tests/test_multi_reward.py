from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_reward_has_progress_component():
    env = GymInspectorEnv(max_steps=4)

    try:
        env.reset(seed=1)

        _, reward_one, _, _, _ = env.step(1)

        env.reset(seed=1)

        _, reward_zero, _, _, _ = env.step(0)

        assert reward_one > reward_zero
    finally:
        env.close()


def test_terminal_reward_is_present():
    env = GymInspectorEnv(max_steps=4)

    try:
        env.reset(seed=1)

        rewards = []

        for _ in range(4):
            _, reward, terminated, truncated, _ = env.step(1)
            rewards.append(reward)
            if terminated or truncated:
                break

        assert len(rewards) == 4
        assert rewards[-1] > rewards[-2]
    finally:
        env.close()

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_action_rewards_are_not_collapsed_to_one_value():
    env = GymInspectorEnv(max_steps=10)

    rewards = []

    for action in [0, 1, 2]:
        env.reset(seed=123)
        _, reward, _, _, _ = env.step(action)
        rewards.append(reward)

    assert all(isinstance(value, float) for value in rewards)
    assert len(set(rewards)) == 3

    env.close()


def test_terminal_reward_has_bonus():
    env = GymInspectorEnv(max_steps=6)
    env.reset(seed=123)

    rewards = []

    for action in [2, 2, 2]:
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    assert terminated is True
    assert truncated is False
    assert rewards[-1] > rewards[0]

    env.close()

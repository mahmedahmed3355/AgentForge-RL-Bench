from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_multiple_reward_values_exist():
    env = GymInspectorEnv(max_steps=5)
    env.reset(seed=1)

    _, positive_reward, _, _, _ = env.step(1)

    env.reset(seed=1)
    _, negative_reward, _, _, _ = env.step(0)

    assert isinstance(positive_reward, float)
    assert isinstance(negative_reward, float)
    assert positive_reward != negative_reward

    env.close()


def test_terminal_reward_is_distinct():
    env = GymInspectorEnv(max_steps=4)
    env.reset(seed=1)

    rewards = []
    for _ in range(4):
        _, reward, terminated, truncated, _ = env.step(1)
        rewards.append(reward)
        if terminated or truncated:
            break

    assert terminated is True
    assert truncated is False
    assert rewards[-1] == 2.0

    env.close()

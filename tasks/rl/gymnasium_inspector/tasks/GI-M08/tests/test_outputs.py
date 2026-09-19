import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_observations_remain_in_space():
    env = GymInspectorEnv(max_steps=7)

    observation, _ = env.reset(seed=123)

    assert env.observation_space.contains(observation)

    for _ in range(6):
        observation, _, terminated, truncated, _ = env.step(1)

        assert env.observation_space.contains(observation)

        if terminated or truncated:
            break

    env.close()


def test_reward_is_float():
    env = GymInspectorEnv()

    env.reset(seed=123)
    _, reward, _, _, _ = env.step(1)

    assert isinstance(reward, float)

    env.close()

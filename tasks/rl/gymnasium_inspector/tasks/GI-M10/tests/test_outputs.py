import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "data"))

from gym_env import GymInspectorEnv


def test_observation_membership():
    env = GymInspectorEnv()

    obs, _ = env.reset(seed=123)

    assert env.observation_space.contains(obs)

    for action in [1, 0, 1]:
        obs, _, terminated, truncated, _ = env.step(action)

        assert env.observation_space.contains(obs)

        if terminated or truncated:
            break

    env.close()


def test_action_membership():
    env = GymInspectorEnv()

    env.reset(seed=123)

    for action in [0, 1]:
        assert env.action_space.contains(action)
        env.step(action)

    env.close()

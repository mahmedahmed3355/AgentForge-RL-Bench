from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_state_upper_bound_is_inclusive():
    env = GymInspectorEnv(max_steps=10)
    env.reset(seed=123)

    for _ in range(5):
        observation, _, terminated, truncated, _ = env.step(2)

        assert int(observation["state"][0]) <= 5

        if terminated or truncated:
            break

    assert terminated is True
    assert int(observation["state"][0]) == 5

    env.close()


def test_horizon_does_not_create_out_of_space_observation():
    env = GymInspectorEnv(max_steps=2)
    env.reset(seed=123)

    observation, _, terminated, truncated, _ = env.step(0)
    assert env.observation_space.contains(observation)

    observation, _, terminated, truncated, _ = env.step(0)

    assert env.observation_space.contains(observation)
    assert terminated is False
    assert truncated is True

    env.close()

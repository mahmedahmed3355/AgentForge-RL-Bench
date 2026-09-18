from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_observation_dtype_matches_space():
    env = GymInspectorEnv(max_steps=6)
    observation, _ = env.reset(seed=1)

    assert observation["state"].dtype == env.observation_space["state"].dtype
    assert observation["steps"].dtype == env.observation_space["steps"].dtype

    env.close()


def test_observation_bounds_match_runtime_state():
    env = GymInspectorEnv(max_steps=6)
    env.reset(seed=1)

    for action in [0, 1, 2, 2]:
        observation, _, terminated, truncated, _ = env.step(action)

        assert 0 <= int(observation["state"][0]) <= 5
        assert 0 <= int(observation["steps"][0]) <= 6

        if terminated or truncated:
            break

    env.close()

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_observation_space_contains_reset_observation():
    env = GymInspectorEnv()

    observation, _ = env.reset(seed=123)

    assert env.observation_space.contains(observation)

    env.close()


def test_observation_space_contains_step_observation():
    env = GymInspectorEnv()

    env.reset(seed=123)

    observation, _, _, _, _ = env.step(1)

    assert env.observation_space.contains(observation)

    env.close()

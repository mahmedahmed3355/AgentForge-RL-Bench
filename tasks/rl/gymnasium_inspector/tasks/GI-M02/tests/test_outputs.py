from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_observation_space_contains_reset_observation():
    env = GymInspectorEnv(max_steps=5)
    observation, _ = env.reset(seed=7)
    assert env.observation_space.contains(observation)
    env.close()


def test_observation_space_contains_step_observation():
    env = GymInspectorEnv(max_steps=5)
    env.reset(seed=7)
    observation, _, _, _, _ = env.step(1)
    assert env.observation_space.contains(observation)
    env.close()

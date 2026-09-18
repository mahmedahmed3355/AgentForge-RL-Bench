from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_success_boundary():
    env = GymInspectorEnv(max_steps=7)

    env.reset(seed=123)

    for _ in range(7):
        observation, reward, terminated, truncated, info = env.step(1)

    assert terminated is True
    assert truncated is False
    assert not (terminated and truncated)
    assert env.observation_space.contains(observation)

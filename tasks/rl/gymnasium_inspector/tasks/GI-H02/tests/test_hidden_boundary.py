import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_terminated_and_truncated_are_mutually_exclusive():
    env = GymInspectorEnv(max_steps=6)
    env.reset(seed=123)

    for _ in range(6):
        _, _, terminated, truncated, _ = env.step(1)
        assert not (terminated and truncated)
        if terminated or truncated:
            break

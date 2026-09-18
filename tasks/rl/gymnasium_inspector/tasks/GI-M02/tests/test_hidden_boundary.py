from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_success_at_horizon_is_termination():
    env = GymInspectorEnv(max_steps=4)
    env.reset(seed=42)

    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(1)

    assert terminated is True
    assert truncated is False

    env.close()


def test_failure_at_horizon_is_truncation():
    env = GymInspectorEnv(max_steps=4)
    env.reset(seed=42)

    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(0)

    assert terminated is False
    assert truncated is True

    env.close()

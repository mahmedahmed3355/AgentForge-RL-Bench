from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_horizon_truncates_without_success():
    env = GymInspectorEnv(max_steps=2)

    env.reset(seed=123)

    _, _, terminated, truncated, _ = env.step(0)
    assert terminated is False
    assert truncated is False

    _, _, terminated, truncated, _ = env.step(0)

    assert terminated is False
    assert truncated is True
    assert not (terminated and truncated)

    env.close()

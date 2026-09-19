import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "data"))

from gym_env import GymInspectorEnv


def test_hidden_boundary_contract():
    env = GymInspectorEnv(max_steps=3)

    env.reset(seed=99)

    for action in [0, 0, 0]:
        _, _, terminated, truncated, _ = env.step(action)

    assert terminated is False
    assert truncated is True
    assert not (terminated and truncated)

    env.close()

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_success_boundary():
    env = GymInspectorEnv(max_steps=4)

    env.reset(seed=1)

    for _ in range(4):
        observation, reward, terminated, truncated, info = env.step(1)

    assert terminated is True
    assert truncated is False
    assert not (terminated and truncated)
    assert info["step"] == 4
    assert info["position"] == 4

    env.close()


def test_truncation_boundary():
    env = GymInspectorEnv(max_steps=4)

    env.reset(seed=1)

    for _ in range(4):
        observation, reward, terminated, truncated, info = env.step(0)

    assert terminated is False
    assert truncated is True
    assert not (terminated and truncated)
    assert info["step"] == 4

    env.close()


def test_post_terminal_step_rejected():
    env = GymInspectorEnv(max_steps=2)

    env.reset(seed=1)
    env.step(1)
    env.step(1)

    try:
        env.step(1)
        assert False
    except RuntimeError:
        pass

    env.close()

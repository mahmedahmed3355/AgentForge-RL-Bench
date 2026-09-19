import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_reward_has_multiple_components():
    env = GymInspectorEnv()

    env.reset(seed=123)
    _, reward_move, _, _, _ = env.step(1)

    env.reset(seed=123)
    _, reward_idle, _, _, _ = env.step(0)

    assert isinstance(reward_move, float)
    assert isinstance(reward_idle, float)
    assert reward_move != reward_idle

    env.close()


def test_terminal_reward_differs_from_nonterminal():
    env = GymInspectorEnv(max_steps=7)

    env.reset(seed=123)

    for _ in range(6):
        _, reward, terminated, truncated, _ = env.step(1)
        assert not terminated
        assert not truncated

    _, terminal_reward, terminated, truncated, _ = env.step(1)

    assert terminated is True
    assert truncated is False
    assert terminal_reward > reward

    env.close()

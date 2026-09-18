import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "data"))

from gym_env import BranchingIrreversibleEnv


def test_reward_progress_component():
    env = BranchingIrreversibleEnv()
    env.reset(seed=123)

    _, reward, _, _, info = env.step(0)

    assert reward == 0.5
    assert info["stage"] == "inspect"


def test_reward_commit_component():
    env = BranchingIrreversibleEnv()
    env.reset(seed=123)
    env.step(0)

    _, reward, _, _, info = env.step(1)

    assert reward == 2.0
    assert info["committed_branch"] == 0


def test_reward_execution_component():
    env = BranchingIrreversibleEnv()
    env.reset(seed=123)
    env.step(0)
    env.step(1)

    _, reward, _, _, info = env.step(2)

    assert reward == 3.0
    assert info["execution"] == "completed"


def test_terminal_reward_component():
    env = BranchingIrreversibleEnv()
    env.reset(seed=123)

    for action in [0, 1, 2]:
        env.step(action)

    _, reward, terminated, truncated, info = env.step(3)

    assert reward == 4.0
    assert terminated is True
    assert truncated is False
    assert info["success"] is True

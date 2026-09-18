import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "data"))

from gym_env import BranchingIrreversibleEnv


def test_trajectory_records_actions():
    env = BranchingIrreversibleEnv()
    env.reset(seed=123)

    for action in [0, 1, 2, 3]:
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    assert len(env.history) == 4

    actions = [entry["action"] for entry in env.history]

    assert actions == [0, 1, 2, 3]


def test_trajectory_states_are_consistent():
    env = BranchingIrreversibleEnv()
    env.reset(seed=123)

    for action in [0, 1, 2, 3]:
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    assert env.history[-1]["terminated"] is True
    assert env.history[-1]["truncated"] is False
    assert env.history[-1]["state"] == 5

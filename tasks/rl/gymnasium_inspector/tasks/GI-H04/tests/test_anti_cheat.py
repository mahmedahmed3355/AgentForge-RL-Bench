import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import DependencyGraphEnv


def test_state_is_not_completed_by_failed_action():
    env = DependencyGraphEnv()

    observation, _ = env.reset(seed=123)

    before = observation["completed"].copy()

    observation, reward, terminated, truncated, info = env.step(5)

    assert reward == -0.5
    assert observation["completed"].tolist() == before.tolist()
    assert observation["progress"] == 0
    assert terminated is False
    assert truncated is False


def test_repeated_completion_does_not_fake_progress():
    env = DependencyGraphEnv()

    env.reset(seed=123)

    env.step(0)

    observation, reward, _, _, info = env.step(0)

    assert reward == -1.0
    assert observation["progress"] == 1
    assert info["valid_action"] is False

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import DependencyGraphEnv


def test_indirect_dependency_is_enforced():
    env = DependencyGraphEnv()
    env.reset(seed=123)

    _, reward, terminated, truncated, info = env.step(4)

    assert reward == -0.5
    assert terminated is False
    assert truncated is False
    assert info["valid_action"] is False


def test_final_node_requires_indirect_chain():
    env = DependencyGraphEnv()
    env.reset(seed=123)

    _, reward, _, _, info = env.step(5)

    assert reward == -0.5
    assert info["valid_action"] is False

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_deterministic_seed_behavior():
    env1 = GymInspectorEnv(max_steps=6)
    env2 = GymInspectorEnv(max_steps=6)

    obs1, info1 = env1.reset(seed=999)
    obs2, info2 = env2.reset(seed=999)

    assert obs1.tolist() == obs2.tolist()
    assert info1 == info2

    trace1 = []
    trace2 = []

    for action in [1, 1, 1, 1, 1, 1]:
        trace1.append(env1.step(action))
        trace2.append(env2.step(action))

    for left, right in zip(trace1, trace2):
        assert left[0].tolist() == right[0].tolist()
        assert left[1:] == right[1:]

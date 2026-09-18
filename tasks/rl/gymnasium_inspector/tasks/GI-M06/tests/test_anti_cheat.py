from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_same_seed_same_trajectory():
    env1 = GymInspectorEnv(max_steps=7)
    env2 = GymInspectorEnv(max_steps=7)

    observation1, info1 = env1.reset(seed=999)
    observation2, info2 = env2.reset(seed=999)

    assert observation1.tolist() == observation2.tolist()
    assert info1 == info2

    for action in [1, 1, 1, 1, 1, 1, 1]:
        result1 = env1.step(action)
        result2 = env2.step(action)

        assert result1[0].tolist() == result2[0].tolist()
        assert result1[1] == result2[1]
        assert result1[2] == result2[2]
        assert result1[3] == result2[3]
        assert result1[4] == result2[4]

        if result1[2] or result1[3]:
            break

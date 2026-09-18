import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_trajectory_step_numbers_are_monotonic():
    env = GymInspectorEnv(max_steps=5)

    env.reset(seed=123)

    records = []

    for action in [1, 1, 1, 1, 1]:
        observation, reward, terminated, truncated, info = env.step(action)

        records.append(
            (
                info["step"],
                info["position"],
                terminated,
                truncated,
            )
        )

        if terminated or truncated:
            break

    assert [r[0] for r in records] == [1, 2, 3, 4, 5]
    assert records[-1][2] is True
    assert records[-1][3] is False

    env.close()


def test_reset_restarts_trajectory():
    env = GymInspectorEnv(max_steps=5)

    env.reset(seed=123)
    env.step(1)
    env.step(1)

    observation, info = env.reset(seed=123)

    assert info["step"] == 0
    assert info["position"] == 0
    assert observation["step"][0] == 0
    assert observation["position"][0] == 0

    env.close()

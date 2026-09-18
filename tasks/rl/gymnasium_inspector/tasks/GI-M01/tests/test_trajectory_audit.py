from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_step_counter_is_monotonic():
    env = GymInspectorEnv(max_steps=4)

    try:
        env.reset(seed=5)

        for expected_step in range(1, 5):
            _, _, terminated, truncated, info = env.step(1)

            assert info["step"] == expected_step

            if terminated or truncated:
                break
    finally:
        env.close()


def test_reset_clears_step_counter():
    env = GymInspectorEnv(max_steps=4)

    try:
        env.reset(seed=5)
        env.step(1)
        env.step(1)

        env.reset(seed=5)

        _, _, _, _, info = env.step(1)

        assert info["step"] == 1
    finally:
        env.close()

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_joint_state_invariant():
    env = GymInspectorEnv(max_steps=6)
    env.reset(seed=123)

    for _ in range(6):
        observation, _, terminated, truncated, info = env.step(1)

        position = int(observation["position"][0])
        energy = int(observation["energy"][0])

        assert position + energy == 6
        assert info["state_consistent"] is True

        if terminated or truncated:
            break

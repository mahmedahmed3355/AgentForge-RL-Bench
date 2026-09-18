from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_trajectory_is_auditable():
    env = GymInspectorEnv(max_steps=8)
    observation, info = env.reset(seed=123)

    trajectory = []

    for step_id in range(8):
        action = 1
        result = env.step(action)
        next_observation, reward, terminated, truncated, step_info = result

        trajectory.append(
            {
                "step": step_id + 1,
                "observation": observation.tolist(),
                "action": action,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "next_observation": next_observation.tolist(),
                "info": dict(step_info),
            }
        )

        observation = next_observation

        if terminated or truncated:
            break

    assert trajectory
    assert trajectory[-1]["terminated"] is True
    assert trajectory[-1]["truncated"] is False

    env.close()

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_trajectory_records_valid_transitions():
    env = GymInspectorEnv(max_steps=6)

    observation, info = env.reset(seed=123)

    trajectory = []

    for step_id in range(6):
        action = 1

        (
            next_observation,
            reward,
            terminated,
            truncated,
            step_info,
        ) = env.step(action)

        trajectory.append(
            {
                "step_id": step_id,
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "terminated": terminated,
                "truncated": truncated,
                "info": step_info,
            }
        )

        assert env.observation_space.contains(next_observation)
        assert env.action_space.contains(action)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(step_info, dict)
        assert not (terminated and truncated)

        observation = next_observation

        if terminated or truncated:
            break

    assert trajectory
    assert trajectory[-1]["terminated"] is True

    env.close()

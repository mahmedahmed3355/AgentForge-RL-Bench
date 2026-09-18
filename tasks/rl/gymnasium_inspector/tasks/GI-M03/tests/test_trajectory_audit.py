from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_trajectory_actions_and_observations_are_space_valid():
    env = GymInspectorEnv(max_steps=6)
    observation, info = env.reset(seed=123)

    trajectory = []

    for action in [0, 1, 2, 1, 0]:
        next_observation, reward, terminated, truncated, info = env.step(action)

        trajectory.append(
            (
                observation,
                action,
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            )
        )

        observation = next_observation

        if terminated or truncated:
            break

    assert trajectory

    for (
        old_observation,
        action,
        new_observation,
        reward,
        terminated,
        truncated,
        info,
    ) in trajectory:
        assert env.observation_space.contains(old_observation)
        assert env.action_space.contains(action)
        assert env.observation_space.contains(new_observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert not (terminated and truncated)

    env.close()

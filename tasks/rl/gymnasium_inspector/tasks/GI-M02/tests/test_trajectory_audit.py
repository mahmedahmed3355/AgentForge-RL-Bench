from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"
sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_trajectory_records_valid_transition_contract():
    env = GymInspectorEnv(max_steps=5)
    observation, info = env.reset(seed=123)

    trajectory = []

    for action in [1, 0, 1, 1]:
        next_observation, reward, terminated, truncated, info = env.step(action)

        trajectory.append(
            {
                "observation": observation,
                "action": action,
                "next_observation": next_observation,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
        )

        observation = next_observation

        if terminated or truncated:
            break

    assert len(trajectory) > 0

    for transition in trajectory:
        assert env.observation_space.contains(transition["observation"])
        assert env.action_space.contains(transition["action"])
        assert env.observation_space.contains(transition["next_observation"])
        assert isinstance(transition["reward"], float)
        assert isinstance(transition["terminated"], bool)
        assert isinstance(transition["truncated"], bool)
        assert isinstance(transition["info"], dict)
        assert not (
            transition["terminated"]
            and transition["truncated"]
        )

    env.close()

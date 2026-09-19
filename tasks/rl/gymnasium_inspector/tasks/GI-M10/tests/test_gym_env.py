import sys
from pathlib import Path

from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "data"))

from gym_env import GymInspectorEnv


def test_gymnasium_checker():
    env = GymInspectorEnv(max_steps=8)
    check_env(env)
    env.close()


def test_reset_contract():
    env = GymInspectorEnv()

    result = env.reset(seed=123)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert env.observation_space.contains(result[0])
    assert isinstance(result[1], dict)

    env.close()


def test_step_contract():
    env = GymInspectorEnv()

    env.reset(seed=123)

    result = env.step(1)

    assert isinstance(result, tuple)
    assert len(result) == 5
    assert env.observation_space.contains(result[0])
    assert isinstance(result[1], float)
    assert isinstance(result[2], bool)
    assert isinstance(result[3], bool)
    assert isinstance(result[4], dict)

    env.close()


def test_seeded_reset_is_deterministic():
    env = GymInspectorEnv()

    obs1, info1 = env.reset(seed=123)
    obs2, info2 = env.reset(seed=123)

    assert (obs1["state"] == obs2["state"]).all()
    assert (obs1["step"] == obs2["step"]).all()
    assert info1 == info2

    env.close()


def test_seeded_trajectory_is_deterministic():
    def run():
        env = GymInspectorEnv(max_steps=8)

        obs, _ = env.reset(seed=123)

        result = [
            int(obs["state"][0]),
            int(obs["step"][0]),
        ]

        for action in [1, 1, 1, 1]:
            obs, reward, terminated, truncated, _ = env.step(action)

            result.append(
                (
                    int(obs["state"][0]),
                    int(obs["step"][0]),
                    float(reward),
                    bool(terminated),
                    bool(truncated),
                )
            )

            if terminated or truncated:
                break

        env.close()

        return result

    assert run() == run()


def test_termination_exclusivity():
    env = GymInspectorEnv(max_steps=8)

    env.reset(seed=123)

    for action in [1] * 8:
        _, _, terminated, truncated, _ = env.step(action)

        assert not (terminated and truncated)

        if terminated or truncated:
            break

    env.close()


def test_horizon_truncates_unsuccessful_episode():
    env = GymInspectorEnv(max_steps=3)

    env.reset(seed=999)

    for _ in range(3):
        _, _, terminated, truncated, _ = env.step(0)

    assert terminated is False
    assert truncated is True
    assert not (terminated and truncated)

    env.close()

from __future__ import annotations

from environment.data.gym_env import AmbiguousFailureDiagnosisEnv


def test_output_types():
    env = AmbiguousFailureDiagnosisEnv()

    obs, info = env.reset(seed=123)

    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)

    obs, reward, terminated, truncated, info = env.step(0)

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

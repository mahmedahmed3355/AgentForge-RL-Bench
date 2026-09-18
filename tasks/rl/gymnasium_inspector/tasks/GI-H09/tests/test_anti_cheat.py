from environment.data.gym_env import ComposedInspectorEnv


def test_private_target_is_not_exposed_in_observation():
    env = ComposedInspectorEnv()
    obs, _ = env.reset(seed=123)

    assert "_target" not in obs
    assert "target" not in obs


def test_public_observation_has_no_hidden_answer():
    env = ComposedInspectorEnv()
    obs, _ = env.reset(seed=123)

    assert obs["diagnosis"] == -1
    assert obs["verified"] == 0

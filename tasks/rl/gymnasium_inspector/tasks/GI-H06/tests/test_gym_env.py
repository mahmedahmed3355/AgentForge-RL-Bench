import numpy as np
from gymnasium.utils.env_checker import check_env
from environment.data.gym_env import ProgressPreservingRecoveryEnv

def test_reset_contract():
    env = ProgressPreservingRecoveryEnv()
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert info["stage"] == "inspect"

def test_seed_determinism():
    a = ProgressPreservingRecoveryEnv()
    b = ProgressPreservingRecoveryEnv()
    oa, ia = a.reset(seed=123)
    ob, ib = b.reset(seed=123)
    assert np.array_equal(oa, ob)
    assert ia == ib

def test_failure_after_two_milestones():
    env = ProgressPreservingRecoveryEnv()
    env.reset(seed=123)
    env.step(1)
    _, _, _, _, info = env.step(1)
    assert info["progress"] == 2
    assert info["failed"] is True

def test_recovery_preserves_progress():
    env = ProgressPreservingRecoveryEnv()
    env.reset(seed=123)
    env.step(1)
    env.step(1)
    _, reward, _, _, info = env.step(2)
    assert reward == 2.5
    assert info["progress"] == 2
    assert info["recovered"] is True

def test_finish_requires_recovery():
    env = ProgressPreservingRecoveryEnv()
    env.reset(seed=123)
    _, reward, terminated, _, info = env.step(3)
    assert terminated is False
    assert reward < 0
    assert info["valid_action"] is False

def test_gymnasium_checker():
    check_env(ProgressPreservingRecoveryEnv(), skip_render_check=True)

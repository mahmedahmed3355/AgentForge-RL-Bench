from environment.data.gym_env import ProgressPreservingRecoveryEnv

def test_recovery_before_failure_is_invalid():
    e = ProgressPreservingRecoveryEnv(); e.reset(seed=123)
    _, r, _, _, info = e.step(2)
    assert info["valid_action"] is False
    assert r < 0
    assert e.progress == 0

def test_recovery_does_not_reset_progress():
    e = ProgressPreservingRecoveryEnv(); e.reset(seed=123)
    e.step(1); e.step(1); e.step(2)
    assert e.progress == 2
    assert e.recovered is True

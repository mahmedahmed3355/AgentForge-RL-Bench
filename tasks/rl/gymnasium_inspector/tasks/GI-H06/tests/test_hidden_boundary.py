from environment.data.gym_env import ProgressPreservingRecoveryEnv

def test_recovery_boundary_preserves_milestones():
    e = ProgressPreservingRecoveryEnv(); e.reset(seed=7)
    e.step(1); e.step(1)
    assert e.progress == 2 and e.failed
    e.step(2)
    assert e.progress == 2 and e.recovered

def test_continue_is_blocked_before_recovery():
    e = ProgressPreservingRecoveryEnv(); e.reset(seed=7)
    e.step(1); e.step(1)
    _, r, _, _, info = e.step(1)
    assert info["valid_action"] is False
    assert r < 0
    assert e.progress == 2

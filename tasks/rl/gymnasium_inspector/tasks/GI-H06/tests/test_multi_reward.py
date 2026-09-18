from environment.data.gym_env import ProgressPreservingRecoveryEnv

def test_progress_reward():
    e = ProgressPreservingRecoveryEnv(); e.reset(seed=123)
    _, r, _, _, _ = e.step(1)
    assert r == 1.0

def test_recovery_reward():
    e = ProgressPreservingRecoveryEnv(); e.reset(seed=123)
    e.step(1); e.step(1)
    _, r, _, _, info = e.step(2)
    assert r == 2.5
    assert info["recovered"] is True

def test_terminal_reward():
    e = ProgressPreservingRecoveryEnv(); e.reset(seed=123)
    e.step(1); e.step(1); e.step(2); e.step(1); e.step(1)
    _, r, term, _, _ = e.step(3)
    assert r == 4.0
    assert term is True

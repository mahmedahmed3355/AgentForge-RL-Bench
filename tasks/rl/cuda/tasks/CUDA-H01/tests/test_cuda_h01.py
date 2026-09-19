from __future__ import annotations

import sys
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TASK_ROOT.parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(TASK_ROOT))

from environment.env import CudaH01Environment
from oracle.oracle import solve
from verifier.verifier import Verifier


def test_deterministic_reset():
    a = CudaH01Environment(seed=17)
    b = CudaH01Environment(seed=17)
    assert a.reset() == b.reset()


def test_different_seeds_change_hidden_scenario():
    a = CudaH01Environment(seed=1)
    b = CudaH01Environment(seed=2)
    a.reset()
    b.reset()
    assert a.workload != b.workload


def test_observation_hides_oracle_truth():
    env = CudaH01Environment(seed=0)
    obs = env.reset()
    assert "bottleneck" not in obs["workload"]
    assert "failure_class" not in obs["workload"]


def test_action_branching_changes_future_state():
    a = CudaH01Environment(seed=3)
    b = CudaH01Environment(seed=3)
    a.reset()
    b.reset()
    a.step({"tool": "modify_transfer_strategy", "args": {"strategy": "async"}})
    b.step({"tool": "modify_transfer_strategy", "args": {"strategy": "sync"}})
    assert a._observation()["config"] != b._observation()["config"]


def test_delayed_failure_is_not_immediately_visible():
    env = CudaH01Environment(seed=0)
    env.reset()
    env.step({"tool": "modify_transfer_strategy", "args": {"strategy": "async"}})
    env.step({"tool": "modify_stream_config", "args": {"streams": 2}})
    env.step({"tool": "validate_output"})
    assert env.state.health == "healthy"
    env.step({"tool": "stress_test"})
    assert "delayed_synchronization_failure" in env.state.detected_failures


def test_recovery_changes_state():
    env = CudaH01Environment(seed=0)
    env.reset()
    env.step({"tool": "modify_transfer_strategy", "args": {"strategy": "async"}})
    env.step({"tool": "modify_stream_config", "args": {"streams": 2}})
    env.step({"tool": "stress_test"})
    env.step({"tool": "modify_event_config", "args": {"events": True}})
    env.step({"tool": "modify_sync_strategy", "args": {"strategy": "event"}})
    env.step({"tool": "rollback_change"})
    assert env.state.recovery_count == 1


def test_noop_baseline_cannot_verify():
    env = CudaH01Environment(seed=0)
    env.reset()
    for _ in range(10):
        env.step({"tool": "inspect_device"})
    env.finalize()
    assert not env.state.success


def test_reward_farming_is_bounded():
    env = CudaH01Environment(seed=0)
    env.reset()
    total = 0.0
    for _ in range(10):
        total += env.step({"tool": "inspect_device"}).reward.total
    assert total < 1.0


def test_oracle_passes():
    result = solve(0)
    assert result["success"]
    assert 70 <= result["steps"] <= 100
    verdict = Verifier().verify(result["trajectory"], result["final_state"])
    assert verdict.passed, verdict.reasons


def test_oracle_generalizes_to_held_out_seeds():
    for seed in (1, 7, 19, 41):
        result = solve(seed)
        assert result["success"], seed
        assert 60 <= result["steps"] <= 100


def test_verifier_rejects_incomplete_behavior():
    verdict = Verifier().verify(
        [{"action": {"tool": "inspect_device"}}],
        {"success": False, "terminal": False},
    )
    assert not verdict.passed


def test_multiple_valid_final_configurations_are_possible():
    for seed in (2, 3, 5):
        env = CudaH01Environment(seed=seed)
        env.reset()
        env.state.config.memory_strategy = "pinned"
        env.state.config.transfer_strategy = "async"
        env.state.config.streams = 2
        env.state.config.events = True
        env.state.config.sync_strategy = "event"
        env.state.config.blocks = 256
        env.state.config.registers = 40
        assert env._performance_score() >= env.workload.required_performance


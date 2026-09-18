from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.data.gym_env import InterruptedEpisodeResumeEnv


def test_interruption_boundary_is_explicit():
    env = InterruptedEpisodeResumeEnv()
    obs, info = env.reset()

    assert info["checkpoint_valid"] is True
    assert info["checkpoint_progress"] == 2
    assert obs["interrupted"] == 1


def test_checkpoint_progress_is_not_zero():
    env = InterruptedEpisodeResumeEnv()
    obs, _ = env.reset()

    assert obs["checkpoint_progress"] == 2


def test_resume_starts_from_checkpoint():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    obs, _, _, _, _ = env.step(1)

    assert obs["progress"] == 2

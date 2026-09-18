from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.data.gym_env import InterruptedEpisodeResumeEnv


def test_duplicate_resume_is_detected():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    env.step(1)
    _, reward, _, _, info = env.step(1)

    assert reward < 0
    assert info["duplicate_transition"] is True


def test_stale_state_is_detected():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    env.step(1)
    env.step(0)
    _, reward, _, _, info = env.step(1)

    assert reward < 0
    assert info["duplicate_transition"] is True


def test_invalid_finish_cannot_fake_success():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    _, reward, terminated, _, info = env.step(3)

    assert reward < 0
    assert terminated is False
    assert info["event"] == "invalid_finish"


def test_terminal_requires_no_duplicate_transition():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    env.step(1)
    env.step(1)

    _, reward, terminated, _, _ = env.step(3)

    assert reward < 0
    assert terminated is False

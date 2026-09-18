from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.data.gym_env import InterruptedEpisodeResumeEnv


def test_inspection_reward():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    _, reward, _, _, _ = env.step(0)

    assert reward == 0.5


def test_resume_reward():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    env.step(0)
    _, reward, _, _, _ = env.step(1)

    assert reward == 2.5


def test_progress_rewards():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    env.step(1)
    _, r1, _, _, _ = env.step(2)
    _, r2, _, _, _ = env.step(2)

    assert r1 == 1.0
    assert r2 == 1.0


def test_terminal_reward():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    env.step(1)
    env.step(2)
    env.step(2)
    _, reward, terminated, _, _ = env.step(3)

    assert reward == 5.0
    assert terminated is True

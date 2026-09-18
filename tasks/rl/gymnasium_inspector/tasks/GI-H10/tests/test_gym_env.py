from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.data.gym_env import InterruptedEpisodeResumeEnv


def test_reset_contract():
    env = InterruptedEpisodeResumeEnv()
    obs, info = env.reset(seed=123)

    assert env.observation_space.contains(obs)
    assert info["checkpoint_progress"] == 2
    assert info["checkpoint_valid"] is True
    assert obs["interrupted"] == 1


def test_seed_determinism():
    env1 = InterruptedEpisodeResumeEnv()
    env2 = InterruptedEpisodeResumeEnv()

    obs1, info1 = env1.reset(seed=123)
    obs2, info2 = env2.reset(seed=123)

    assert obs1 == obs2
    assert info1 == info2


def test_gymnasium_checker():
    env = InterruptedEpisodeResumeEnv()
    check_env(env, skip_render_check=True)


def test_checkpoint_is_persistent():
    env = InterruptedEpisodeResumeEnv()
    obs, info = env.reset()

    assert obs["checkpoint_progress"] == 2
    assert info["persistent_progress"] == 2


def test_resume_restores_checkpoint():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    env.step(0)
    obs, _, _, _, info = env.step(1)

    assert obs["progress"] == 2
    assert obs["resumed"] == 1
    assert info["event"] == "checkpoint_resumed"


def test_resume_cannot_be_replayed():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    env.step(1)
    obs, reward, _, _, info = env.step(1)

    assert reward < 0
    assert info["event"] == "duplicate_resume"
    assert info["duplicate_transition"] is True


def test_continue_before_resume_is_invalid():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    _, reward, _, _, info = env.step(2)

    assert reward < 0
    assert info["event"] == "continue_before_resume"


def test_finish_requires_complete_state():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    _, reward, terminated, _, info = env.step(3)

    assert reward < 0
    assert terminated is False
    assert info["event"] == "invalid_finish"


def test_terminal_state():
    env = InterruptedEpisodeResumeEnv()
    env.reset()

    env.step(1)
    env.step(2)
    env.step(2)
    obs, reward, terminated, truncated, info = env.step(3)

    assert terminated is True
    assert truncated is False
    assert reward == 5.0
    assert obs["progress"] == 4
    assert obs["finished"] == 1
    assert info["duplicate_transition"] is False
    assert info["stale_state"] is False


def test_observation_space():
    env = InterruptedEpisodeResumeEnv()
    obs, _ = env.reset()

    assert env.observation_space.contains(obs)
    assert isinstance(obs["progress"], (int, np.integer))

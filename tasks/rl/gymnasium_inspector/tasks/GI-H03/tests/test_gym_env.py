from pathlib import Path
import sys

import gymnasium as gym
from gymnasium.utils.env_checker import check_env

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "data"))

from gym_env import BranchingIrreversibleEnv


def test_reset_contract():
    env = BranchingIrreversibleEnv()
    result = env.reset(seed=1)

    assert isinstance(result, tuple)
    assert len(result) == 2

    observation, info = result

    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)


def test_step_contract():
    env = BranchingIrreversibleEnv()
    env.reset(seed=1)

    result = env.step(0)

    assert isinstance(result, tuple)
    assert len(result) == 5

    observation, reward, terminated, truncated, info = result

    assert env.observation_space.contains(observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_space():
    env = BranchingIrreversibleEnv()

    assert env.action_space.contains(0)
    assert env.action_space.contains(1)
    assert env.action_space.contains(2)
    assert env.action_space.contains(3)


def test_observation_space():
    env = BranchingIrreversibleEnv()
    observation, _ = env.reset(seed=7)

    assert env.observation_space.contains(observation)


def test_gymnasium_checker():
    env = BranchingIrreversibleEnv()
    check_env(env, skip_render_check=True)


def test_successful_terminal_semantics():
    env = BranchingIrreversibleEnv(max_steps=10)
    env.reset(seed=123)

    terminated = False
    truncated = False

    for action in [0, 1, 2, 3]:
        _, _, terminated, truncated, _ = env.step(action)

    assert terminated is True
    assert truncated is False
    assert not (terminated and truncated)


def test_horizon_truncation():
    env = BranchingIrreversibleEnv(max_steps=2)
    env.reset(seed=123)

    _, _, terminated, truncated, _ = env.step(0)
    assert not terminated
    assert not truncated

    _, _, terminated, truncated, _ = env.step(0)

    assert terminated is False
    assert truncated is True
    assert not (terminated and truncated)


def test_irreversible_commitment():
    env = BranchingIrreversibleEnv()
    env.reset(seed=123)

    env.step(0)
    _, _, _, _, info = env.step(1)

    assert info["irreversible"] is True

    _, reward, _, _, info = env.step(1)

    assert reward == -1.0
    assert info["error"] == "commitment_is_irreversible"


def test_wrong_execution_order():
    env = BranchingIrreversibleEnv()
    env.reset(seed=123)

    _, reward, _, _, info = env.step(2)

    assert reward == -1.0
    assert info["error"] == "must_commit_before_execute"

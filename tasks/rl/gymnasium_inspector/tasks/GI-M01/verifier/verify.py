from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))


def _fail(reason: str) -> int:
    result = {
        "passed": False,
        "reason": reason,
    }

    print(json.dumps(result, indent=2))
    return 1


def _contains_valid_observation(env: Any, observation: Any) -> bool:
    return env.observation_space.contains(observation)


def main() -> int:
    try:
        module = importlib.import_module("gym_env")
        env_cls = getattr(module, "GymInspectorEnv")
    except Exception as exc:
        return _fail(f"import failure: {exc}")

    try:
        env = env_cls(max_steps=4)

        observation, info = env.reset(seed=123)

        if not isinstance(info, dict):
            return _fail("reset info is not a dictionary")

        if not _contains_valid_observation(env, observation):
            return _fail("reset observation is outside observation_space")

        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        for action in [1, 1, 1, 1]:
            if not env.action_space.contains(action):
                return _fail("oracle action outside action_space")

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            if not isinstance(reward, float):
                return _fail("reward is not float")

            if not isinstance(terminated, bool):
                return _fail("terminated is not bool")

            if not isinstance(truncated, bool):
                return _fail("truncated is not bool")

            if not isinstance(info, dict):
                return _fail("step info is not dictionary")

            if not _contains_valid_observation(env, observation):
                return _fail("step observation is outside observation_space")

            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        if not terminated:
            return _fail("intended trajectory did not terminate successfully")

        if truncated:
            return _fail("successful trajectory was truncated")

        if int(observation["position"][0]) != 4:
            return _fail("final position is incorrect")

        if steps != 4:
            return _fail("unexpected trajectory length")

        first_reset, _ = env.reset(seed=999)
        second_reset, _ = env.reset(seed=999)

        if not np_equal_observation(first_reset, second_reset):
            return _fail("seeded resets are not deterministic")

        if not env.action_space.contains(0):
            return _fail("declared action_space is invalid")

        if not env.action_space.contains(1):
            return _fail("declared action_space is invalid")

        if not env.action_space.contains(2):
            return _fail("declared action_space is invalid")

        env.close()

        result = {
            "passed": True,
            "steps": steps,
            "reward_total": float(total_reward),
            "final_position": 4,
            "deterministic_reset": True,
        }

        print(json.dumps(result, indent=2))
        return 0

    except Exception as exc:
        return _fail(f"verification failure: {exc}")


def np_equal_observation(left: Any, right: Any) -> bool:
    return (
        left.keys() == right.keys()
        and all(
            (left[key] == right[key]).all()
            for key in left
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())

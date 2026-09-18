from __future__ import annotations

import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


SEED = 123
MAX_SEARCH_STEPS = 8


def run_sequence(actions):
    env = GymInspectorEnv(max_steps=MAX_SEARCH_STEPS)

    try:
        observation, info = env.reset(seed=SEED)

        rewards = []
        terminated = False
        truncated = False

        for action in actions:
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            rewards.append(float(reward))

            if terminated or truncated:
                break

        return {
            "success": bool(terminated and not truncated),
            "steps": len(rewards),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "observation": observation,
            "reward_total": float(sum(rewards)),
            "rewards": rewards,
        }
    finally:
        env.close()


def discover_action_values():
    env = GymInspectorEnv(max_steps=MAX_SEARCH_STEPS)

    try:
        action_space = env.action_space

        if hasattr(action_space, "n"):
            return list(range(int(action_space.n)))

        raise RuntimeError(
            "GI-M03 oracle requires a finite discrete action space"
        )
    finally:
        env.close()


def main() -> int:
    action_values = discover_action_values()

    successful_result = None
    successful_actions = None

    for length in range(1, MAX_SEARCH_STEPS + 1):
        for actions in itertools.product(action_values, repeat=length):
            result = run_sequence(actions)

            if result["success"]:
                successful_result = result
                successful_actions = list(actions)
                break

        if successful_result is not None:
            break

    if successful_result is None:
        result = {
            "success": False,
            "steps": 0,
            "terminated": False,
            "truncated": False,
            "actions": [],
            "reward_total": 0.0,
        }
        return_code = 1
    else:
        final_observation = successful_result["observation"]

        final_state = None
        if isinstance(final_observation, dict):
            if "state" in final_observation:
                state_value = final_observation["state"]
                try:
                    final_state = int(state_value[0])
                except (TypeError, ValueError, IndexError):
                    try:
                        final_state = int(state_value)
                    except (TypeError, ValueError):
                        final_state = None

            elif "position" in final_observation:
                position_value = final_observation["position"]
                try:
                    final_state = int(position_value[0])
                except (TypeError, ValueError, IndexError):
                    try:
                        final_state = int(position_value)
                    except (TypeError, ValueError):
                        final_state = None

        result = {
            "success": True,
            "steps": successful_result["steps"],
            "terminated": successful_result["terminated"],
            "truncated": successful_result["truncated"],
            "actions": successful_actions,
            "final_state": final_state,
            "reward_total": successful_result["reward_total"],
            "rewards": successful_result["rewards"],
            "observation_valid": True,
        }

        return_code = 0

    output = ROOT / "oracle_result.json"
    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())

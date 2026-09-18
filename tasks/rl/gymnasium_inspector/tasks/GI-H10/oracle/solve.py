from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from environment.data.gym_env import InterruptedEpisodeResumeEnv


def main() -> None:
    env = InterruptedEpisodeResumeEnv()

    obs, info = env.reset(seed=2026)
    trajectory = []

    # Inspect checkpoint.
    obs, reward, terminated, truncated, info = env.step(0)
    trajectory.append(
        {
            "action": 0,
            "progress": obs["progress"],
            "checkpoint_progress": obs["checkpoint_progress"],
            "resumed": obs["resumed"],
            "event": info["event"],
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
    )

    # Resume from the persistent checkpoint exactly once.
    obs, reward, terminated, truncated, info = env.step(1)
    trajectory.append(
        {
            "action": 1,
            "progress": obs["progress"],
            "checkpoint_progress": obs["checkpoint_progress"],
            "resumed": obs["resumed"],
            "event": info["event"],
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
    )

    # Continue only remaining work.
    for _ in range(2):
        obs, reward, terminated, truncated, info = env.step(2)
        trajectory.append(
            {
                "action": 2,
                "progress": obs["progress"],
                "checkpoint_progress": obs["checkpoint_progress"],
                "resumed": obs["resumed"],
                "event": info["event"],
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            }
        )

    obs, reward, terminated, truncated, info = env.step(3)
    trajectory.append(
        {
            "action": 3,
            "progress": obs["progress"],
            "checkpoint_progress": obs["checkpoint_progress"],
            "resumed": obs["resumed"],
            "committed": obs["committed"],
            "finished": obs["finished"],
            "duplicate_transition": info["duplicate_transition"],
            "stale_state": info["stale_state"],
            "event": info["event"],
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
    )

    reward_total = sum(float(x["reward"]) for x in trajectory)

    result = {
        "success": bool(
            terminated
            and obs["progress"] == 4
            and obs["checkpoint_progress"] == 2
            and obs["resumed"] == 1
            and obs["committed"] == 1
            and obs["finished"] == 1
            and info["duplicate_transition"] is False
            and info["stale_state"] is False
        ),
        "steps": len(trajectory),
        "terminated": terminated,
        "truncated": truncated,
        "final_progress": int(obs["progress"]),
        "final_checkpoint_progress": int(obs["checkpoint_progress"]),
        "final_resumed": bool(obs["resumed"]),
        "final_committed": bool(obs["committed"]),
        "final_finished": bool(obs["finished"]),
        "duplicate_transition": bool(info["duplicate_transition"]),
        "stale_state": bool(info["stale_state"]),
        "reward_total": reward_total,
        "observation_valid": env.observation_space.contains(obs),
        "trajectory_length": len(trajectory),
        "trajectory": trajectory,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

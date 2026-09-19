import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "data"))

from gym_env import GymInspectorEnv


def test_trajectory_audit():
    env = GymInspectorEnv(max_steps=8)

    obs, info = env.reset(seed=123)

    records = []

    for action in [1, 1, 1, 1, 1, 1]:
        (
            next_obs,
            reward,
            terminated,
            truncated,
            next_info,
        ) = env.step(action)

        records.append(
            {
                "action": int(action),
                "state": int(next_obs["state"][0]),
                "step": int(next_obs["step"][0]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

        assert int(next_obs["step"][0]) == len(records)
        assert not (terminated and truncated)

        if terminated or truncated:
            break

    assert len(records) > 0
    assert records[-1]["terminated"] is True
    assert records[-1]["truncated"] is False

    env.close()

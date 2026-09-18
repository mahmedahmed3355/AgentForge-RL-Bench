from pathlib import Path
import sys

from gymnasium.utils.env_checker import check_env


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "environment" / "data"

sys.path.insert(0, str(DATA))

from gym_env import GymInspectorEnv


def test_gymnasium_checker():
    env = GymInspectorEnv()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()

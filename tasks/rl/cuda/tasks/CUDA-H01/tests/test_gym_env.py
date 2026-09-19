from environment.gym_env import CudaH01GymEnv


def test_gym_adapter_reset_and_step():
    env = CudaH01GymEnv(seed=0)
    obs, info = env.reset()
    assert "step" in obs
    assert "task_id" in info
    obs, reward, terminated, truncated, info = env.step(0)
    assert "step" in obs
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["task_id"] == "CUDA-H01"
    env.close()


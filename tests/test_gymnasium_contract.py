import gymnasium as gym

from agentforge.core import RewardBreakdown
from agentforge.environments import (
    AgentForgeGymEnv,
    BaseEnvironment,
    EnvironmentStep,
    RLEnvironmentAdapter,
)


class TinyLegacyEnvironment(BaseEnvironment):
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        return {"state": "start"}

    def step(self, action):
        return EnvironmentStep(
            observation={"state": "done"},
            reward=RewardBreakdown(
                terminal=1.0
            ),
            done=True,
            info={"action": action},
        )

    def close(self):
        pass


def test_rl_adapter_is_a_real_gymnasium_env():
    env = RLEnvironmentAdapter(
        TinyLegacyEnvironment()
    )

    assert isinstance(env, gym.Env)

    observation, info = env.reset(
        seed=123
    )

    assert observation == {
        "state": "start"
    }
    assert info["seed"] == 123

    result = env.step(
        {"type": "finish"}
    )

    assert result[0] == {
        "state": "done"
    }
    assert result[1] == 1.0
    assert result[2] is True
    assert result[3] is False
    assert result[4]["action"] == {
        "type": "finish"
    }


def test_gymnasium_native_contract_has_reset_and_step():
    assert issubclass(
        AgentForgeGymEnv,
        gym.Env,
    )

    assert hasattr(
        AgentForgeGymEnv,
        "reset",
    )

    assert hasattr(
        AgentForgeGymEnv,
        "step",
    )

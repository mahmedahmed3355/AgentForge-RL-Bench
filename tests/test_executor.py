from agentforge.core.executor import EpisodeExecutor
from agentforge.environments import MockBackendEnvironment


def test_executor_runs_complete_successful_episode():
    env = MockBackendEnvironment()

    def agent(observation):
        return "solve"

    executor = EpisodeExecutor(
        environment=env,
        agent=agent,
        max_steps=100,
    )

    result = executor.run(
        episode_id="episode-executor-001",
        task_id="backend-001",
    )

    assert result.success is True
    assert result.steps == 1
    assert result.final_status == "success"
    assert result.total_reward == 1.7

    env.close()


def test_executor_runs_multi_step_episode():
    env = MockBackendEnvironment(max_steps=3)

    def agent(observation):
        if observation["step"] < 2:
            return "work"
        return "solve"

    executor = EpisodeExecutor(
        environment=env,
        agent=agent,
        max_steps=100,
    )

    result = executor.run(
        episode_id="episode-executor-002",
        task_id="backend-002",
    )

    assert result.success is True
    assert result.steps == 3
    assert result.final_status == "success"

    env.close()


def test_executor_respects_environment_timeout():
    env = MockBackendEnvironment(max_steps=2)

    def agent(observation):
        return "work"

    executor = EpisodeExecutor(
        environment=env,
        agent=agent,
        max_steps=100,
    )

    result = executor.run(
        episode_id="episode-executor-003",
        task_id="backend-003",
    )

    assert result.success is False
    assert result.steps == 2
    assert result.final_status == "timeout"

    env.close()

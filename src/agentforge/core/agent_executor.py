"""Agent and environment integration for AgentForge-RL-Bench."""

from __future__ import annotations

from dataclasses import dataclass

from agentforge.agents.base import BaseAgent
from agentforge.environments import BaseEnvironment

from .episode import EpisodeRunner


@dataclass
class AgentEpisodeResult:
    """Result of one agent-environment episode."""

    episode_id: str
    task_id: str
    steps: int
    done: bool
    total_reward: float
    trajectory_length: int


class AgentEpisodeExecutor:
    """Execute an Agent inside an AgentForge environment."""

    def __init__(
        self,
        agent: BaseAgent,
        environment: BaseEnvironment,
    ) -> None:
        self.agent = agent
        self.environment = environment

    def run(
        self,
        episode_id: str,
        task_id: str,
        max_steps: int,
    ) -> AgentEpisodeResult:
        """Run one complete agent-environment episode."""

        self.agent.reset()

        reset_result = self.environment.reset()
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            observation, _ = reset_result
        else:
            observation = reset_result

        runner = EpisodeRunner(
            episode_id=episode_id,
            task_id=task_id,
            max_steps=max_steps,
        )

        for _ in range(max_steps):
            action = self.agent.act(observation)

            transition = self.environment.step(action)

            runner.record_step(
                observation=observation,
                action=action,
                reward=transition.reward,
                next_observation=transition.observation,
                status=(
                    "success"
                    if transition.done
                    else "running"
                ),
            )

            self.agent.update(
                observation=observation,
                action=action,
                reward=transition.reward.total,
                next_observation=transition.observation,
                done=transition.done,
            )

            observation = transition.observation

            if transition.done:
                break

        trajectory = runner.recorder.trajectory

        # `done` in the executor reflects environment termination.
        # Reaching max_steps alone is a bounded execution, not
        # necessarily successful environment termination.
        environment_done = (
            trajectory.length > 0
            and trajectory.steps[-1].status == "success"
        )

        return AgentEpisodeResult(
            episode_id=episode_id,
            task_id=task_id,
            steps=runner.step_count,
            done=environment_done,
            total_reward=trajectory.total_reward,
            trajectory_length=trajectory.length,
        )

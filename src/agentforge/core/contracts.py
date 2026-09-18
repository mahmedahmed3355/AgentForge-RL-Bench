"""Core data contracts for AgentForge-RL-Bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Domain = Literal["backend", "cuda", "distributed"]
Split = Literal["train", "eval"]
StepStatus = Literal["running", "success", "failure", "timeout"]


@dataclass
class TaskSpec:
    """Immutable-style description of a benchmark task."""

    task_id: str
    domain: Domain
    split: Split
    description: str
    difficulty: str
    version: str = "1.0"


@dataclass
class RewardBreakdown:
    """Detailed reward components for one environment transition."""

    progress: float = 0.0
    correctness: float = 0.0
    tests: float = 0.0
    efficiency: float = 0.0
    terminal: float = 0.0
    penalties: float = 0.0

    @property
    def total(self) -> float:
        """Return the total scalar reward."""

        return (
            self.progress
            + self.correctness
            + self.tests
            + self.efficiency
            + self.terminal
            + self.penalties
        )


@dataclass
class StepRecord:
    """One agent-environment interaction."""

    step_id: int
    observation: Any
    action: Any
    reward: RewardBreakdown
    next_observation: Any
    status: StepStatus
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Complete result of one attempt on one task."""

    episode_id: str
    task_id: str
    success: bool
    steps: int
    total_reward: float
    final_status: StepStatus
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Ordered collection of all steps from an episode."""

    episode_id: str
    task_id: str
    steps: list[StepRecord] = field(default_factory=list)

    def append(self, step: StepRecord) -> None:
        """Append one step to the trajectory."""

        self.steps.append(step)

    @property
    def length(self) -> int:
        """Number of recorded steps."""

        return len(self.steps)

    @property
    def total_reward(self) -> float:
        """Sum all step rewards."""

        return sum(step.reward.total for step in self.steps)


@dataclass
class CheckpointRef:
    """Reference to a saved agent state."""

    checkpoint_id: str
    agent_id: str
    path: str
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating an agent on a task split."""

    evaluation_id: str
    agent_id: str
    split: Split
    task_results: list[EpisodeResult] = field(default_factory=list)

    @property
    def task_count(self) -> int:
        return len(self.task_results)

    @property
    def passed_count(self) -> int:
        return sum(result.success for result in self.task_results)

    @property
    def pass_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return self.passed_count / self.task_count


@dataclass
class ExperimentResult:
    """Top-level record for a complete AgentForge experiment."""

    experiment_id: str
    agent_id: str
    training_tasks: list[str]
    evaluation: EvaluationResult
    checkpoint: CheckpointRef | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

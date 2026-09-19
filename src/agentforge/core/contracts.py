"""Core benchmark contracts for AgentForge-RL-Bench."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


Domain = Literal["backend", "cuda", "distributed"]
Split = Literal["train", "eval"]
StepStatus = Literal["running", "success", "failure", "timeout"]


@dataclass
class SeedConfig:
    """Deterministic configuration for benchmark execution."""

    base_seed: int
    deterministic: bool = True
    namespace: str = "agentforge"

    def __post_init__(self) -> None:
        if not isinstance(self.base_seed, int):
            raise TypeError("base_seed must be an integer.")

        if self.base_seed < 0:
            raise ValueError("base_seed must be >= 0.")

        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be a bool.")

        if not self.namespace.strip():
            raise ValueError("namespace must not be empty.")


@dataclass
class TaskSpec:
    """Validated description of a benchmark task."""

    task_id: str
    domain: Domain
    split: Split
    description: str
    difficulty: str
    version: str = "1.0"
    capabilities: tuple[str, ...] = ()
    seed_config: SeedConfig = field(
        default_factory=lambda: SeedConfig(base_seed=0)
    )

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id must not be empty.")

        if not self.version.strip():
            raise ValueError("version must not be empty.")

        if not self.description.strip():
            raise ValueError("description must not be empty.")

        if not self.difficulty.strip():
            raise ValueError("difficulty must not be empty.")

        if self.domain not in {
            "backend",
            "cuda",
            "distributed",
        }:
            raise ValueError(
                f"Unsupported domain: {self.domain!r}"
            )

        if self.split not in {"train", "eval"}:
            raise ValueError(
                f"Unsupported split: {self.split!r}"
            )

        if not isinstance(self.seed_config, SeedConfig):
            raise TypeError(
                "seed_config must be a SeedConfig."
            )

        normalized = tuple(
            value.strip()
            for value in self.capabilities
            if value.strip()
        )

        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "capabilities must not contain duplicates."
            )

        object.__setattr__(
            self,
            "capabilities",
            normalized,
        )

    @property
    def identity(self) -> str:
        """Stable versioned task identity."""

        return f"{self.task_id}@{self.version}"

    @property
    def tags(self) -> tuple[str, ...]:
        """Compatibility alias for capability tags."""

        return self.capabilities


class TrajectoryEventType(str, Enum):
    """Explicit benchmark trajectory semantics."""

    DECISION = "decision"
    BRANCH = "branch"
    WRONG_BRANCH = "wrong_branch"
    FAILURE = "failure"
    RECOVERY = "recovery"
    DELAYED_CONSEQUENCE = "delayed_consequence"
    TERMINAL_OUTCOME = "terminal_outcome"


@dataclass(frozen=True)
class TrajectoryEvent:
    """Typed semantic event attached to a trajectory step."""

    step_id: int
    event_type: TrajectoryEventType
    description: str = ""
    branch_id: str | None = None
    related_step_id: int | None = None
    consequence_delay: int | None = None
    severity: str | None = None
    evidence: str | None = None

    def __post_init__(self) -> None:
        if self.step_id < 1:
            raise ValueError("step_id must be >= 1.")

        if (
            self.event_type
            in {
                TrajectoryEventType.BRANCH,
                TrajectoryEventType.WRONG_BRANCH,
            }
            and not self.branch_id
        ):
            raise ValueError(
                "Branch events require branch_id."
            )

        if (
            self.event_type
            == TrajectoryEventType.DELAYED_CONSEQUENCE
            and self.consequence_delay is None
        ):
            raise ValueError(
                "Delayed consequence events require "
                "consequence_delay."
            )

        if (
            self.consequence_delay is not None
            and self.consequence_delay < 0
        ):
            raise ValueError(
                "consequence_delay must be >= 0."
            )


@dataclass(frozen=True)
class DecisionEvent(TrajectoryEvent):
    """Agent decision point."""

    event_type: Literal[TrajectoryEventType.DECISION] = (
        TrajectoryEventType.DECISION
    )


@dataclass(frozen=True)
class BranchEvent(TrajectoryEvent):
    """Selected branch."""

    event_type: Literal[TrajectoryEventType.BRANCH] = (
        TrajectoryEventType.BRANCH
    )


@dataclass(frozen=True)
class WrongBranchEvent(TrajectoryEvent):
    """Wrong or suboptimal branch."""

    event_type: Literal[TrajectoryEventType.WRONG_BRANCH] = (
        TrajectoryEventType.WRONG_BRANCH
    )


@dataclass(frozen=True)
class FailureEvent(TrajectoryEvent):
    """Failure occurrence."""

    event_type: Literal[TrajectoryEventType.FAILURE] = (
        TrajectoryEventType.FAILURE
    )


@dataclass(frozen=True)
class RecoveryEvent(TrajectoryEvent):
    """Recovery from a previous failure."""

    event_type: Literal[TrajectoryEventType.RECOVERY] = (
        TrajectoryEventType.RECOVERY
    )


@dataclass(frozen=True)
class DelayedConsequenceEvent(TrajectoryEvent):
    """Delayed consequence of an earlier action."""

    event_type: Literal[
        TrajectoryEventType.DELAYED_CONSEQUENCE
    ] = TrajectoryEventType.DELAYED_CONSEQUENCE


@dataclass(frozen=True)
class TerminalOutcomeEvent(TrajectoryEvent):
    """Terminal episode outcome."""

    event_type: Literal[
        TrajectoryEventType.TERMINAL_OUTCOME
    ] = TrajectoryEventType.TERMINAL_OUTCOME


TrajectoryEventRecord = (
    DecisionEvent
    | BranchEvent
    | WrongBranchEvent
    | FailureEvent
    | RecoveryEvent
    | DelayedConsequenceEvent
    | TerminalOutcomeEvent
)


@dataclass(frozen=True)
class RewardProvenance:
    """Attribution explaining where a reward component came from."""

    source: str
    component: str
    value: float
    reason: str = ""
    step_id: int | None = None
    event_type: TrajectoryEventType | None = None
    evidence: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError(
                "Reward provenance source must not be empty."
            )

        if not self.component.strip():
            raise ValueError(
                "Reward provenance component must not be empty."
            )


# Backward-compatible name for callers that use the longer form.
RewardAttribution = RewardProvenance


@dataclass
class RewardBreakdown:
    """Multi-component reward contract."""

    progress: float = 0.0
    correctness: float = 0.0
    tests: float = 0.0
    recovery: float = 0.0
    efficiency: float = 0.0
    terminal: float = 0.0
    penalties: float = 0.0
    provenance: tuple[RewardProvenance, ...] = ()

    @property
    def testing(self) -> float:
        """Canonical semantic alias for the existing tests field."""

        return self.tests

    @property
    def total(self) -> float:
        """Return the scalar sum of all reward components."""

        return (
            self.progress
            + self.correctness
            + self.tests
            + self.recovery
            + self.efficiency
            + self.terminal
            + self.penalties
        )


@dataclass
class StepRecord:
    """One complete agent-environment interaction."""

    step_id: int
    observation: Any
    action: Any
    reward: RewardBreakdown
    next_observation: Any
    status: StepStatus
    events: list[TrajectoryEventRecord] = field(
        default_factory=list
    )
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
        self.steps.append(step)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def total_reward(self) -> float:
        return sum(
            step.reward.total
            for step in self.steps
        )


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
    """Result of evaluating an agent on one task split."""

    evaluation_id: str
    agent_id: str
    split: Split
    task_results: list[EpisodeResult] = field(
        default_factory=list
    )

    @property
    def task_count(self) -> int:
        return len(self.task_results)

    @property
    def passed_count(self) -> int:
        return sum(
            result.success
            for result in self.task_results
        )

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

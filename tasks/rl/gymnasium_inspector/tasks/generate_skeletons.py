from __future__ import annotations

from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parent
SPEC = ROOT.parent / "specification"
OUT = ROOT

MATRIX = SPEC / "CAPABILITY_MATRIX_V1.yaml"
CARDS = SPEC / "DESIGN_CARDS_V1.yaml"


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_cards(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("tasks", "design_cards", "cards", "entries"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(f"Unsupported design-card structure: {type(data).__name__}")


def normalize_matrix(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("tasks", "matrix", "entries"):
            value = data.get(key)
            if isinstance(value, list):
                return value

    raise ValueError(f"Unsupported capability-matrix structure: {type(data).__name__}")


def get_value(item, *keys, default=None):
    for key in keys:
        if isinstance(item, dict) and key in item:
            return item[key]
    return default


def task_id(item, index):
    value = get_value(
        item,
        "task_id",
        "id",
        "name",
        default=f"gymnasium-inspector-{index:03d}",
    )
    return str(value)


def difficulty(item):
    value = get_value(item, "difficulty", "level", default="medium")
    return str(value).lower().replace("_", " ")


def slug(value):
    return (
        str(value)
        .lower()
        .replace(" ", "-")
        .replace("_", "-")
    )


def write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_task(card, matrix_item, index):
    tid = task_id(card, index)
    level = difficulty(card)

    capability = get_value(
        card,
        "capability",
        "primary_capability",
        default=get_value(matrix_item, "capability", default="inspection"),
    )

    traps = get_value(
        card,
        "traps",
        "reasoning_traps",
        default=[],
    )

    if not isinstance(traps, list):
        traps = [str(traps)]

    environment = get_value(
        card,
        "environment",
        "environment_type",
        default="gymnasium_inspector",
    )

    horizon = get_value(
        card,
        "horizon",
        "max_steps",
        default=100,
    )

    reward_components = [
        "progress",
        "correctness",
        "tests",
        "efficiency",
        "terminal",
        "penalties",
    ]

    task_dir = OUT / tid

    task_toml = f'''schema_version = "1.4"

[task]
id = "{tid}"
name = "{tid}"
difficulty = "{level}"
domain = "rl"
family = "gymnasium_inspector"

authors = [
  {{ name = "AgentForge", email = "agentforge@example.com" }}
]

[metadata]
capability = "{capability}"
environment = "{environment}"
horizon = {horizon}
oracle_required = true
verifier_required = true
hidden_evaluation = true
multi_reward = true

[reward]
components = ["progress", "correctness", "tests", "efficiency", "terminal", "penalties"]
aggregation = "RewardBreakdown.total"

[environment]
family = "gymnasium_inspector"
stateful = true
partial_observability = true
deterministic_seed = true
checkpointable = true

[oracle]
enabled = true
source_of_truth = "reference_behavior"
modifies_candidate = false
writes_reference_artifacts = false

[verifier]
enabled = true
validates_final_state = true
validates_trajectory = true
validates_hidden_cases = true
rejects_reward_hacking = true

[termination]
success = "goal_state_verified"
failure = "invalid_terminal_state"
timeout = "max_steps"
'''

    instruction = f'''# Task: {tid}

## Objective

Solve the Gymnasium/Inspector environment task and reach the verified target state.

## Domain

RL / Gymnasium / Inspector

## Difficulty

{level}

## Primary capability

{capability}

## Environment

The environment is stateful and interaction-driven.

The agent must use observations and available actions/tools to determine the correct sequence of operations.

The target state is not assumed to be directly visible.

## Constraints

- Do not assume hidden state.
- Do not bypass the environment.
- Do not modify verifier or oracle artifacts.
- Do not rely on a fixed reference trajectory.
- The final state must satisfy the verifier.
- Intermediate actions must remain valid environment interactions.

## Reasoning requirements

The task is evaluated over an interaction trajectory rather than a single answer.

The agent may need to:

1. Inspect the current state.
2. Select an appropriate action.
3. Observe the resulting transition.
4. Update its plan.
5. Recover from incorrect intermediate decisions when possible.
6. Reach the final target state.
7. Produce behavior accepted by the verifier.

## Hidden evaluation

The evaluator may use unseen parameterizations and adversarial cases.

Visible task information must not expose the complete reference trajectory.

## Reward

The environment exposes multiple reward components:

- progress
- correctness
- tests
- efficiency
- terminal
- penalties

The final reward is the sum represented by the environment's RewardBreakdown contract.

Reward must reflect correct behavior rather than shortcut exploitation.

## Success

Success requires verified final-state correctness and a valid interaction trajectory.
'''

    env_py = f'''from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentforge.environments.base import BaseEnvironment, EnvironmentStep
from agentforge.core import RewardBreakdown


@dataclass
class {slug(tid).replace("-", "_").title().replace("_", "")}Environment(BaseEnvironment):
    """Stateful Gymnasium/Inspector skeleton for {tid}."""

    seed: int | None = None
    max_steps: int = {horizon}

    def __post_init__(self) -> None:
        self._step_count = 0
        self._state: dict[str, Any] = {{}}
        self._done = False

    def reset(self) -> dict[str, Any]:
        self._step_count = 0
        self._done = False
        self._state = {{
            "task_id": "{tid}",
            "phase": "initial",
            "progress": 0.0,
        }}
        return dict(self._state)

    def step(self, action: Any) -> EnvironmentStep:
        if self._done:
            raise RuntimeError("Cannot step a finished environment.")

        self._step_count += 1

        next_state = dict(self._state)

        reward = RewardBreakdown(
            progress=0.0,
            correctness=0.0,
            tests=0.0,
            efficiency=0.0,
            terminal=0.0,
            penalties=0.0,
        )

        done = self._step_count >= self.max_steps

        if done:
            self._done = True

        self._state = next_state

        return EnvironmentStep(
            observation=dict(next_state),
            reward=reward,
            done=done,
            info={{
                "task_id": "{tid}",
                "step": self._step_count,
            }},
        )

    def close(self) -> None:
        self._done = True
'''

    oracle_py = f'''from __future__ import annotations

from typing import Any


class Oracle:
    """Reference behavior for {tid}.

    The Oracle is independent from the candidate agent and verifier.
    It must solve the environment through legitimate environment interactions.
    """

    task_id = "{tid}"

    def solve(self, environment: Any) -> list[Any]:
        trajectory: list[Any] = []

        observation = environment.reset()

        for _ in range(getattr(environment, "max_steps", {horizon})):
            action = self.select_action(observation)
            trajectory.append(action)

            transition = environment.step(action)
            observation = transition.observation

            if transition.done:
                break

        return trajectory

    def select_action(self, observation: Any) -> Any:
        raise NotImplementedError(
            "Implement reference behavior for {tid}."
        )
'''

    verifier_py = f'''from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    score: float
    reasons: tuple[str, ...]


class Verifier:
    """Independent verifier for {tid}."""

    task_id = "{tid}"

    def verify(
        self,
        trajectory: Any,
        final_state: Any,
    ) -> VerificationResult:
        reasons: list[str] = []

        if trajectory is None:
            reasons.append("missing_trajectory")

        if final_state is None:
            reasons.append("missing_final_state")

        if reasons:
            return VerificationResult(
                passed=False,
                score=0.0,
                reasons=tuple(reasons),
            )

        # TODO:
        # 1. Validate trajectory transitions.
        # 2. Validate required state invariants.
        # 3. Validate hidden/adversarial conditions.
        # 4. Detect reward-hacking shortcuts.
        # 5. Validate terminal goal state.

        return VerificationResult(
            passed=False,
            score=0.0,
            reasons=("verifier_not_implemented",),
        )
'''

    tests_py = f'''from __future__ import annotations


def test_task_identity():
    assert "{tid}" != ""


def test_task_difficulty():
    assert "{level}" in {{"medium", "hard", "very hard"}}


def test_multi_reward_contract():
    components = {{
        "progress",
        "correctness",
        "tests",
        "efficiency",
        "terminal",
        "penalties",
    }}

    assert len(components) == 6


def test_oracle_required():
    assert True


def test_verifier_required():
    assert True
'''

    metadata = f'''task_id: "{tid}"
domain: "rl"
family: "gymnasium_inspector"
difficulty: "{level}"
capability: "{capability}"
environment: "{environment}"
horizon: {horizon}
oracle: true
verifier: true
hidden_evaluation: true
multi_reward: true
reward_components:
  - progress
  - correctness
  - tests
  - efficiency
  - terminal
  - penalties
reasoning_traps:
{chr(10).join(f"  - {slug(x)}" for x in traps) if traps else "  - none_declared"}
status: "skeleton"
'''

    readme = f'''# {tid}

Status: skeleton

Difficulty: {level}

Primary capability: {capability}

Environment family: Gymnasium / Inspector

This directory is the implementation scaffold for the task.

Required implementation layers:

- environment
- observation
- action/tool interface
- state transitions
- multi-component reward
- constraints
- checkpoint behavior
- termination
- oracle
- verifier
- visible tests
- hidden evaluation
- adversarial validation
'''

    write(task_dir / "task.toml", task_toml)
    write(task_dir / "instruction.md", instruction)
    write(task_dir / "metadata.yaml", metadata)
    write(task_dir / "README.md", readme)
    write(task_dir / "environment" / "__init__.py", "")
    write(task_dir / "environment" / "env.py", env_py)
    write(task_dir / "oracle" / "__init__.py", "")
    write(task_dir / "oracle" / "oracle.py", oracle_py)
    write(task_dir / "verifier" / "__init__.py", "")
    write(task_dir / "verifier" / "verifier.py", verifier_py)
    write(task_dir / "tests" / "__init__.py", "")
    write(task_dir / "tests" / "test_contract.py", tests_py)
    write(task_dir / "tests" / "hidden_placeholder.py", "")
    write(task_dir / "solution" / ".gitkeep", "")

    return tid


def main():
    matrix = normalize_matrix(load_yaml(MATRIX))
    cards = normalize_cards(load_yaml(CARDS))

    if len(cards) != 33:
        raise RuntimeError(f"Expected 33 design cards, found {len(cards)}")

    if len(matrix) != 33:
        raise RuntimeError(f"Expected 33 matrix entries, found {len(matrix)}")

    generated = []

    for index, card in enumerate(cards, start=1):
        matrix_item = matrix[index - 1]
        generated.append(build_task(card, matrix_item, index))

    manifest = {
        "version": "1",
        "family": "gymnasium_inspector",
        "task_count": len(generated),
        "tasks": generated,
        "difficulty_counts": {
            "medium": sum(
                1
                for card in cards
                if difficulty(card) == "medium"
            ),
            "hard": sum(
                1
                for card in cards
                if difficulty(card) == "hard"
            ),
            "very_hard": sum(
                1
                for card in cards
                if difficulty(card) == "very hard"
            ),
        },
        "reward_contract": [
            "progress",
            "correctness",
            "tests",
            "efficiency",
            "terminal",
            "penalties",
        ],
        "status": "skeleton",
    }

    write(
        SPEC / "TASK_SKELETON_MANIFEST_V1.yaml",
        yaml.safe_dump(
            manifest,
            sort_keys=False,
            allow_unicode=True,
        ),
    )

    print("=== GYMNASIUM / INSPECTOR TASK SKELETONS CREATED ===")
    print(f"Tasks: {len(generated)}")
    print("Expected: 33")
    print("Medium:", manifest["difficulty_counts"]["medium"])
    print("Hard:", manifest["difficulty_counts"]["hard"])
    print("Very Hard:", manifest["difficulty_counts"]["very_hard"])
    print("Multi-reward: progress/correctness/tests/efficiency/terminal/penalties")
    print("Oracle: scaffolded")
    print("Verifier: scaffolded")
    print("Hidden evaluation: reserved")
    print("Adversarial validation: reserved")
    print("=== TASK SKELETON GENERATION COMPLETE ===")


if __name__ == "__main__":
    main()

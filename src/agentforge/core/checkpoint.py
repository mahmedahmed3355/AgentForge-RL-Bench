"""Checkpoint persistence for AgentForge-RL-Bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import (
    DecisionEvent,
    BranchEvent,
    WrongBranchEvent,
    FailureEvent,
    RecoveryEvent,
    DelayedConsequenceEvent,
    TerminalOutcomeEvent,
    RewardProvenance,
    RewardBreakdown,
    TrajectoryEventType,
)
from .episode import EpisodeRunner


_EVENT_CLASSES = {
    TrajectoryEventType.DECISION: DecisionEvent,
    TrajectoryEventType.BRANCH: BranchEvent,
    TrajectoryEventType.WRONG_BRANCH: WrongBranchEvent,
    TrajectoryEventType.FAILURE: FailureEvent,
    TrajectoryEventType.RECOVERY: RecoveryEvent,
    TrajectoryEventType.DELAYED_CONSEQUENCE: DelayedConsequenceEvent,
    TrajectoryEventType.TERMINAL_OUTCOME: TerminalOutcomeEvent,
}


class EpisodeCheckpoint:
    """Save and restore an episode's execution state."""

    @staticmethod
    def save(
        runner: EpisodeRunner,
        path: str | Path,
    ) -> Path:
        """Persist current episode state to JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        trajectory = runner.recorder.trajectory

        payload: dict[str, Any] = {
            "episode_id": runner.episode_id,
            "task_id": runner.task_id,
            "max_steps": runner.max_steps,
            "metadata": runner.metadata,
            "finished": runner.done,
            "success": runner._success,
            "final_status": runner._final_status,
            "steps": [
                {
                    "step_id": step.step_id,
                    "observation": step.observation,
                    "action": step.action,
                    "reward": {
                        "progress": step.reward.progress,
                        "correctness": step.reward.correctness,
                        "tests": step.reward.tests,
                        "recovery": step.reward.recovery,
                        "efficiency": step.reward.efficiency,
                        "terminal": step.reward.terminal,
                        "penalties": step.reward.penalties,
                        "provenance": [
                            {
                                "source": item.source,
                                "component": item.component,
                                "value": item.value,
                                "reason": item.reason,
                                "step_id": item.step_id,
                                "event_type": (
                                    item.event_type.value
                                    if item.event_type is not None
                                    else None
                                ),
                            }
                            for item in step.reward.provenance
                        ],
                    },
                    "next_observation": step.next_observation,
                    "status": step.status,
                    "events": [
                        {
                            "event_type": event.event_type.value,
                            "step_id": event.step_id,
                            "description": event.description,
                            "branch_id": event.branch_id,
                            "related_step_id": event.related_step_id,
                            "consequence_delay": event.consequence_delay,
                            "severity": event.severity,
                            "evidence": event.evidence,
                        }
                        for event in step.events
                    ],
                    "metadata": step.metadata,
                }
                for step in trajectory.steps
            ],
        }

        output_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return output_path

    @staticmethod
    def load(path: str | Path) -> EpisodeRunner:
        """Restore an EpisodeRunner from a checkpoint."""

        checkpoint_path = Path(path)

        payload = json.loads(
            checkpoint_path.read_text(
                encoding="utf-8"
            )
        )

        runner = EpisodeRunner(
            episode_id=payload["episode_id"],
            task_id=payload["task_id"],
            max_steps=payload["max_steps"],
            metadata=payload.get("metadata", {}),
        )

        for item in payload["steps"]:
            reward_data = item["reward"]

            provenance = tuple(
                RewardProvenance(
                    source=entry["source"],
                    component=entry["component"],
                    value=entry["value"],
                    reason=entry.get("reason", ""),
                    step_id=entry.get("step_id"),
                    event_type=(
                        TrajectoryEventType(
                            entry["event_type"]
                        )
                        if entry.get("event_type")
                        else None
                    ),
                )
                for entry in reward_data.get(
                    "provenance",
                    [],
                )
            )

            reward = RewardBreakdown(
                progress=reward_data["progress"],
                correctness=reward_data["correctness"],
                tests=reward_data["tests"],
                recovery=reward_data.get(
                    "recovery",
                    0.0,
                ),
                efficiency=reward_data["efficiency"],
                terminal=reward_data["terminal"],
                penalties=reward_data["penalties"],
                provenance=provenance,
            )

            events = []

            for event_data in item.get("events", []):
                event_type = TrajectoryEventType(
                    event_data["event_type"]
                )

                event_class = _EVENT_CLASSES[event_type]

                events.append(
                    event_class(
                        step_id=event_data["step_id"],
                        description=event_data.get(
                            "description",
                            "",
                        ),
                        branch_id=event_data.get(
                            "branch_id"
                        ),
                        related_step_id=event_data.get(
                            "related_step_id"
                        ),
                        consequence_delay=event_data.get(
                            "consequence_delay"
                        ),
                        severity=event_data.get(
                            "severity"
                        ),
                        evidence=event_data.get(
                            "evidence"
                        ),
                    )
                )

            runner.recorder.record_step(
                observation=item["observation"],
                action=item["action"],
                reward=reward,
                next_observation=item["next_observation"],
                status=item["status"],
                events=events,
                metadata=item.get("metadata", {}),
            )

        runner._finished = payload["finished"]
        runner._success = payload["success"]
        runner._final_status = payload["final_status"]

        return runner

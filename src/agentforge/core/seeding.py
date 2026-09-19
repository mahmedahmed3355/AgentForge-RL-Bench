"""Deterministic benchmark seeding utilities."""

from __future__ import annotations

import hashlib

from .contracts import SeedConfig, TaskSpec


def derive_seed(
    config: SeedConfig,
    *,
    task_id: str,
    episode_index: int = 0,
    environment_index: int = 0,
) -> int:
    """Derive a reproducible 32-bit seed for one environment instance."""

    if episode_index < 0:
        raise ValueError("episode_index must be >= 0.")

    if environment_index < 0:
        raise ValueError(
            "environment_index must be >= 0."
        )

    material = (
        f"{config.namespace}|"
        f"{config.base_seed}|"
        f"{task_id}|"
        f"{episode_index}|"
        f"{environment_index}"
    ).encode("utf-8")

    digest = hashlib.sha256(material).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="big",
        signed=False,
    )


def task_seed(
    task: TaskSpec,
    *,
    episode_index: int = 0,
    environment_index: int = 0,
) -> int:
    """Derive the reproducible seed configured for a task."""

    return derive_seed(
        task.seed_config,
        task_id=task.task_id,
        episode_index=episode_index,
        environment_index=environment_index,
    )

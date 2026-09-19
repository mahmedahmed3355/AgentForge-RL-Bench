
from agentforge.core import (
    SeedConfig,
    TaskSpec,
    derive_seed,
    task_seed,
)


def make_task():
    return TaskSpec(
        task_id="seed-task",
        domain="cuda",
        split="train",
        description="Seed contract test.",
        difficulty="easy",
        seed_config=SeedConfig(base_seed=42),
    )


def test_seed_is_reproducible():
    config = SeedConfig(base_seed=42)

    a = derive_seed(
        config,
        task_id="task-a",
        episode_index=3,
        environment_index=2,
    )
    b = derive_seed(
        config,
        task_id="task-a",
        episode_index=3,
        environment_index=2,
    )

    assert a == b


def test_seed_changes_with_episode():
    config = SeedConfig(base_seed=42)

    a = derive_seed(
        config,
        task_id="task-a",
        episode_index=0,
    )
    b = derive_seed(
        config,
        task_id="task-a",
        episode_index=1,
    )

    assert a != b


def test_task_seed_uses_task_seed_configuration():
    task = make_task()

    assert task_seed(
        task,
        episode_index=5,
    ) == task_seed(
        task,
        episode_index=5,
    )

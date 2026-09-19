from agentforge.core import (
    SeedConfig,
    TaskSpec,
    derive_seed,
    task_seed,
)


def make_task():
    return TaskSpec(
        task_id="seed-test",
        domain="backend",
        split="train",
        description="Seed test.",
        difficulty="easy",
        seed_config=SeedConfig(
            base_seed=42
        ),
    )


def test_derived_seed_is_reproducible():
    config = SeedConfig(
        base_seed=42
    )

    first = derive_seed(
        config,
        task_id="task-a",
        episode_index=3,
        environment_index=2,
    )

    second = derive_seed(
        config,
        task_id="task-a",
        episode_index=3,
        environment_index=2,
    )

    assert first == second


def test_different_episode_indices_produce_distinct_streams():
    config = SeedConfig(
        base_seed=42
    )

    first = derive_seed(
        config,
        task_id="task-a",
        episode_index=0,
    )

    second = derive_seed(
        config,
        task_id="task-a",
        episode_index=1,
    )

    assert first != second


def test_task_seed_uses_task_configuration():
    task = make_task()

    assert task_seed(
        task,
        episode_index=4,
    ) == task_seed(
        task,
        episode_index=4,
    )

from agentforge.core import (
    SeedConfig,
    TaskRegistry,
    TaskSpec,
)


def make_task(
    task_id="backend-001",
    version="1.0",
    split="train",
):
    return TaskSpec(
        task_id=task_id,
        version=version,
        domain="backend",
        split=split,
        description="Contract test task.",
        difficulty="medium",
        capabilities=(
            "branching",
            "recovery",
        ),
        seed_config=SeedConfig(
            base_seed=123
        ),
    )


def test_task_spec_identity_and_validation():
    task = make_task()

    assert task.identity == "backend-001@1.0"
    assert task.capabilities == (
        "branching",
        "recovery",
    )


def test_task_registry_register_and_get():
    registry = TaskRegistry()

    task = make_task()

    registry = registry.register(task)

    assert len(registry) == 1
    assert registry.get("backend-001") == task
    assert registry.get(
        "backend-001",
        "1.0",
    ) == task


def test_task_registry_split_filtering():
    registry = TaskRegistry(
        tasks=(
            make_task(
                task_id="train-001",
                split="train",
            ),
            make_task(
                task_id="eval-001",
                split="eval",
            ),
        )
    )

    assert [
        task.task_id
        for task in registry.for_split("train")
    ] == ["train-001"]

    assert [
        task.task_id
        for task in registry.for_split("eval")
    ] == ["eval-001"]


def test_task_registry_rejects_duplicate_identity():
    task = make_task()

    registry = TaskRegistry(
        tasks=(task,)
    )

    try:
        registry.register(task)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError(
            "Duplicate task identity was accepted."
        )

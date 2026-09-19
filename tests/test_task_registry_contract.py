
from agentforge.core import SeedConfig, TaskRegistry, TaskSpec


def make_task(task_id="task-001", version="1.0", split="train"):
    return TaskSpec(
        task_id=task_id,
        domain="backend",
        split=split,
        description="Contract test task.",
        difficulty="medium",
        capabilities=("branching", "recovery"),
        seed_config=SeedConfig(base_seed=123),
        version=version,
    )


def test_task_identity_is_stable():
    task = make_task()
    assert task.identity == "task-001@1.0"
    assert task.tags == ("branching", "recovery")


def test_registry_register_get_and_split():
    registry = TaskRegistry(
        tasks=(
            make_task("train-001", split="train"),
            make_task("eval-001", split="eval"),
        )
    )

    assert registry.get("train-001") .identity == "train-001@1.0"
    assert registry.get("eval-001", "1.0").split == "eval"
    assert [
        task.task_id
        for task in registry.for_split("train")
    ] == ["train-001"]


def test_registry_rejects_duplicate_identity():
    registry = TaskRegistry(tasks=(make_task(),))

    try:
        registry.register(make_task())
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("Duplicate identity was accepted.")


def test_task_validation_rejects_invalid_split():
    try:
        make_task(split="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid split was accepted.")

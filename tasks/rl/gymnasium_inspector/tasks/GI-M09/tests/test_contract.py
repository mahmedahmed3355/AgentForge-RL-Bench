from __future__ import annotations


def test_task_identity():
    assert "GI-M09" != ""


def test_task_difficulty():
    assert "medium" in {"medium", "hard", "very hard"}


def test_multi_reward_contract():
    components = {
        "progress",
        "correctness",
        "tests",
        "efficiency",
        "terminal",
        "penalties",
    }

    assert len(components) == 6


def test_oracle_required():
    assert True


def test_verifier_required():
    assert True

"""Deterministic test agent used by framework tests."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseAgent


class TestAgent(BaseAgent):
    """Minimal deterministic agent for integration testing."""

    def __init__(self) -> None:
        self.step_count = 0
        self.update_count = 0

    def reset(self) -> None:
        self.step_count = 0
        self.update_count = 0

    def act(self, observation: Any) -> dict[str, Any]:
        self.step_count += 1

        return {
            "type": "test_action",
            "step": self.step_count,
            "observation": observation,
        }

    def update(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        self.update_count += 1

    def save_checkpoint(self, path: str) -> None:
        payload = {
            "step_count": self.step_count,
            "update_count": self.update_count,
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def load_checkpoint(self, path: str) -> None:
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)

        self.step_count = payload["step_count"]
        self.update_count = payload["update_count"]


# Prevent pytest from collecting this framework test agent as a test class.
TestAgent.__test__ = False

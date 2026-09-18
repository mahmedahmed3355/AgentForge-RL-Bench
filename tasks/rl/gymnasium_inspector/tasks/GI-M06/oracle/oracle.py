from __future__ import annotations

from typing import Any


class Oracle:
    """Reference behavior for GI-M06.

    The Oracle is independent from the candidate agent and verifier.
    It must solve the environment through legitimate environment interactions.
    """

    task_id = "GI-M06"

    def solve(self, environment: Any) -> list[Any]:
        trajectory: list[Any] = []

        observation = environment.reset()

        for _ in range(getattr(environment, "max_steps", 100)):
            action = self.select_action(observation)
            trajectory.append(action)

            transition = environment.step(action)
            observation = transition.observation

            if transition.done:
                break

        return trajectory

    def select_action(self, observation: Any) -> Any:
        raise NotImplementedError(
            "Implement reference behavior for GI-M06."
        )

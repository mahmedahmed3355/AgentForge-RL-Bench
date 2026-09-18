"""Public adapter for integrating external agents with AgentForge."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .base import BaseAgent


class ExternalAgent(Protocol):
    """Minimal protocol required from an external agent."""

    def reset(self) -> None:
        ...

    def act(self, observation: Any) -> Any:
        ...

    def update(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        ...


class AgentAdapter(BaseAgent):
    """Adapt an external agent to the AgentForge BaseAgent contract."""

    def __init__(self, external_agent: ExternalAgent) -> None:
        self.external_agent = external_agent
        self._checkpoint_state: dict[str, Any] = {}

    def reset(self) -> None:
        """Reset the wrapped external agent."""

        self.external_agent.reset()

    def act(self, observation: Any) -> Any:
        """Request an action from the wrapped external agent."""

        return self.external_agent.act(observation)

    def update(
        self,
        observation: Any,
        action: Any,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> None:
        """Forward a transition to the wrapped external agent."""

        self.external_agent.update(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
        )

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save the external agent checkpoint when supported."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_method = getattr(
            self.external_agent,
            "save_checkpoint",
            None,
        )

        if callable(save_method):
            result = save_method(output_path)

            if result is None:
                return output_path

            return Path(result)

        import json

        output_path.write_text(
            json.dumps(
                {
                    "adapter_state": self._checkpoint_state,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return output_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load the external agent checkpoint when supported."""

        input_path = Path(path)

        load_method = getattr(
            self.external_agent,
            "load_checkpoint",
            None,
        )

        if callable(load_method):
            load_method(input_path)
            return

        import json

        payload = json.loads(
            input_path.read_text(encoding="utf-8")
        )

        self._checkpoint_state = payload.get(
            "adapter_state",
            {},
        )

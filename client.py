# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Automathreasoner Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .env.models import AutomathreasonerAction, AutomathreasonerObservation


class AutomathreasonerEnv(
    EnvClient[AutomathreasonerAction, AutomathreasonerObservation, State]
):
    """
    Client for the Automathreasoner Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with AutomathreasonerEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(AutomathreasonerAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = AutomathreasonerEnv.from_docker_image("AutoMathReasoner-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(AutomathreasonerAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: AutomathreasonerAction) -> Dict:
        """
        Convert AutomathreasonerAction to JSON payload for step message.

        Args:
            action: AutomathreasonerAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "reasoning": action.reasoning,
            "final_answer": action.final_answer,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AutomathreasonerObservation]:
        """
        Parse server response into StepResult[AutomathreasonerObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with AutomathreasonerObservation
        """
        obs_data = payload.get("observation", {})
        observation = AutomathreasonerObservation(
            problem_text=obs_data.get("problem_text", ""),
            difficulty_level=obs_data.get("difficulty_level", 1.0),
            history=obs_data.get("history", []),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

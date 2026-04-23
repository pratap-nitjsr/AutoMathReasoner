# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the AutoMathReasoner Environment.
"""

from typing import List, Dict, Any
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

class AutomathreasonerAction(Action):
    """Action for the AutoMathReasoner environment - containing reasoning and final answer."""

    reasoning: str = Field(..., description="The step-by-step mathematical reasoning.")
    final_answer: str = Field(..., description="The final numerical or algebraic answer.")


class AutomathreasonerObservation(Observation):
    """Observation from the AutoMathReasoner environment."""

    problem_text: str = Field(default="", description="The text of the generated math problem.")
    difficulty_level: float = Field(default=1.0, description="The current difficulty level of the problem.")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="History of the last 3 attempts for this problem.")
    
    # Required by OpenEnv base class
    reward: float = Field(default=0.0, description="Reward received from the previous action.")
    done: bool = Field(default=False, description="Whether the episode has ended.")

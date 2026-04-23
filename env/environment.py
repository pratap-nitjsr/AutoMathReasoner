import logging
from uuid import uuid4
from collections import deque
from typing import Dict, Any, List

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from .models import AutomathreasonerAction, AutomathreasonerObservation
    from .generator import TaskGenerationEngine
    from .verifier import VerifierSystem
    from .rewards import RewardSystem
except ImportError:
    from env.models import AutomathreasonerAction, AutomathreasonerObservation
    from env.generator import TaskGenerationEngine
    from env.verifier import VerifierSystem
    from env.rewards import RewardSystem

logger = logging.getLogger(__name__)

class AutomathreasonerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.generator = TaskGenerationEngine()
        self.verifier = VerifierSystem()
        self.reward_system = RewardSystem(max_len=2000)
        
        # Curriculum tracking
        self.difficulty_level = 2.0  # Starting difficulty
        self.rolling_results = deque(maxlen=20) # Keep track of last 20 results (1 for correct, 0 for incorrect)
        
        # Current problem state
        self.current_problem = ""
        self.current_solution = ""
        self.current_sympy_f = None  # Integration Ground Truth
        self.times_seen_problem = 0
        self.history: List[Dict[str, Any]] = []
        self.max_steps = 3

    def _update_curriculum(self):
        """Update difficulty based on rolling accuracy"""
        if len(self.rolling_results) >= 5:
            accuracy = sum(self.rolling_results) / len(self.rolling_results)
            if accuracy > 0.7:
                self.difficulty_level += 0.5
            elif accuracy < 0.6:
                self.difficulty_level = max(1.0, self.difficulty_level - 0.5)
            logger.info(f"Curriculum Updated: Accuracy={accuracy:.2f}, New Difficulty={self.difficulty_level}")

    def reset(self) -> AutomathreasonerObservation:
        """Reset environment to a new problem."""
        self._update_curriculum()
        
        self._state = State(episode_id=str(uuid4()), step_count=0)
        task = self.generator.generate_task(target_difficulty_band=self.difficulty_level)
        
        self.current_problem = task['problem']
        self.current_solution = task['solution']
        self.current_sympy_f = task.get('sympy_f')
        # The generator returns its own continuous difficulty score; we'll expose the target difficulty band
        self.times_seen_problem = 0
        self.history = []
        
        return AutomathreasonerObservation(
            problem_text=self.current_problem,
            difficulty_level=self.difficulty_level,
            history=[],
            reward=0.0,
            done=False
        )

    def step(self, action: AutomathreasonerAction) -> AutomathreasonerObservation:  # type: ignore[override]
        self._state.step_count += 1
        
        # Verification
        c, q, p_sup, r_ref = self.verifier.verify(
            action.reasoning, 
            action.final_answer, 
            self.current_solution,
            sympy_f=self.current_sympy_f
        )
        
        # Reward
        action_str = f"{action.reasoning} \n {action.final_answer}"
        total_r, components = self.reward_system.compute_reward(
            correctness=c,
            reasoning_quality=q,
            process_supervision=p_sup,
            reflection_score=r_ref,
            action_str=action_str,
            final_answer=action.final_answer,
            history=self.history,
            times_seen_problem=self.times_seen_problem
        )
        
        self.times_seen_problem += 1
        
        # Update history
        attempt = {
            "prediction": action.final_answer,
            "correctness": c
        }
        self.history.append(attempt)
        # Keep only last 3 attempts for observation
        obs_history = self.history[-3:]
        
        is_correct = (c == 1.0)
        done = is_correct or self._state.step_count >= self.max_steps
        
        if done:
            self.rolling_results.append(1 if is_correct else 0)
            
        return AutomathreasonerObservation(
            problem_text=self.current_problem,
            difficulty_level=self.difficulty_level,
            history=obs_history,
            reward=total_r,
            done=done,
            metadata={
                "reward_components": components,
                "ground_truth": self.current_solution if done else "HIDDEN", # Only reveal on done or not at all
                "is_correct": is_correct
            }
        )

    @property
    def state(self) -> State:
        return self._state

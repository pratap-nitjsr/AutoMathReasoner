import random
import math
from typing import Dict, Any, List, Tuple

class RewardSystem:
    def __init__(self, max_len: int = 1000):
        self.max_len = max_len

    def compute_diversity(self, current_answer: str, history: List[Dict[str, Any]]) -> float:
        """
        D = diversity (difference from past attempts)
        If repeated answer, returns a steep exponential penalty: D = -exp(1.0).
        Otherwise, returns D = 1.0.
        """
        if not history:
            return 1.0
            
        cur_ans_clean = current_answer.strip().lower()
        
        for attempt in history:
            prev_ans = attempt.get('final_answer', '').strip().lower()
            if prev_ans == cur_ans_clean:
                return -math.exp(1.0) # Approx -2.71steep penalty
                
        # If unique, give full diversity bonus
        return 1.0

    def compute_efficiency(self, action_string: str) -> float:
        """
        E = efficiency. We use a Gaussian penalty curve:
        E = exp(- (len_ratio)^2 ) - 1
        This smoothly penalizes overly verbose answers.
        """
        approx_tokens = len(action_string) / 4.0
        optimal_tokens = 50.0  # Assumed ideal length
        
        # Ratio mapping constraint
        ratio = (approx_tokens - optimal_tokens) / optimal_tokens
        
        # Smooth gaussian-like decay towards -1.0
        e = math.exp(- (ratio ** 2)) - 1.0
        return e
        
    def compute_exploration_bonus(self, action_string: str, times_seen: int) -> float:
        """
        [PAPER TRACEABILITY: Exploration via Entropy Bonus]
        G. EXPLORATION VIA ENTROPY BONUS
        Computes output diversity (token variance) and adds bonus.
        X = (entropy_bonus) / sqrt(1 + times_seen_problem)
        """
        # Simple structural entropy estimation (unique character distribution variance)
        length = len(action_string)
        if length > 0:
            unique_ratio = len(set(action_string)) / length
            entropy_bonus = math.log1p(unique_ratio)  # Non-linear scaling
        else:
            entropy_bonus = 0.0
            
        return entropy_bonus / math.sqrt(1.0 + times_seen)

    def detect_trivial_output(self, action_string: str) -> bool:
        """Anti-reward hacking: detect trivial constant outputs"""
        # If the output is just a single character repeated or very low entropy
        if len(action_string) < 2:
            return True
        unique_chars = len(set(action_string))
        if unique_chars < 3 and len(action_string) > 10:
            return True
        return False

    def compute_reward(self, 
                      correctness: float, 
                      reasoning_quality: float,
                      process_supervision: float,
                      reflection_score: float,
                      action_str: str, 
                      final_answer: str,
                      history: List[Dict[str, Any]],
                      times_seen_problem: int) -> Tuple[float, Dict[str, float]]:
        """
        [PAPER TRACEABILITY: DeepSeekMath-inspired reward composite]
        R = 0.4*C + 0.2*Q_smooth + 0.15*D + 0.1*E + 0.1*P + 0.1*R + 0.15*X + noise
        """
        if self.detect_trivial_output(action_str):
            # Anti-hacking strongly penalized
            components = {"C": 0.0, "Q": 0.0, "D": 0.0, "E": -1.0, "X": 0.0, "noise": 0.0}
            return -1.0, components
            
        c = correctness
        q = reasoning_quality
        d = self.compute_diversity(final_answer, history)
        
        # If repeated answer, C is zeroed to prevent hacking
        if d < 0:
            c = 0.0
            
        e = self.compute_efficiency(action_str)
        x = self.compute_exploration_bonus(action_str, times_seen_problem)
        
        noise = random.gauss(0, 0.05)
        
        # Smoothly squish reasoning quality using tanh to bound its impact
        q_smooth = math.tanh(q)
        
        # New Composite Reward Equation
        total_r = (0.35 * c) + (0.15 * q_smooth) + (0.1 * process_supervision) + (0.1 * reflection_score) + (0.15 * d) + (0.05 * e) + (0.1 * x) + noise
        
        components = {
            "total_reward": total_r,
            "C_correctness": c,
            "Q_reasoning": q_smooth,
            "P_process_supervision": process_supervision,
            "R_reflection": reflection_score,
            "D_diversity": d,
            "E_efficiency": e,
            "X_exploration": x,
            "noise": noise
        }
        
        return total_r, components

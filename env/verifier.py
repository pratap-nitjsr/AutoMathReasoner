import re
import math
from typing import Dict, Any, Tuple

class VerifierSystem:
    def __init__(self):
        pass

    def check_exact_match(self, prediction: str, ground_truth: str) -> bool:
        """1. Exact match verifier"""
        return prediction.strip().lower() == ground_truth.strip().lower()

    def check_numeric_tolerance(self, prediction: str, ground_truth: str, tol: float = 1e-4) -> bool:
        """2. Numeric tolerance checker"""
        try:
            pred_val = float(prediction.strip())
            gt_val = float(ground_truth.strip())
            return math.isclose(pred_val, gt_val, rel_tol=tol, abs_tol=tol)
        except ValueError:
            return False

    def check_python_execution(self, prediction: str, ground_truth: str) -> bool:
        """3. Python execution (eval safe expressions)"""
        # If prediction is an expression like "2+3", try evaluating it safely
        safe_dict = {"__builtins__": None, "math": math}
        try:
            # We are verifying if evaluating the prediction gives ground truth
            pred_eval = eval(prediction.strip(), safe_dict, {})
            try:
                gt_eval = float(ground_truth.strip())
                return math.isclose(float(pred_eval), gt_eval, rel_tol=1e-4, abs_tol=1e-4)
            except ValueError:
                return str(pred_eval).strip().lower() == ground_truth.strip().lower()
        except Exception:
            return False

    def mock_llm_judge(self, reasoning: str, prediction: str, ground_truth: str) -> float:
        """4. LLM judge (mock or placeholder scoring reasoning quality)
        Returns reasoning quality score Q (0.0 to 1.0)
        """
        # A simple heuristic for mock judge:
        # Longer reasoning with step-like markers suggests higher quality in this mock
        step_markers = ['step', 'first', 'then', 'because', 'therefore', 'equals', '=', '+', '-']
        score = 0.0
        
        # Length bonus (up to 0.4)
        length = len(reasoning.split())
        score += min(0.4, length * 0.01)
        
        # Structure bonus (up to 0.6)
        lower_reasoning = reasoning.lower()
        marker_count = sum(1 for m in step_markers if m in lower_reasoning)
        score += min(0.6, marker_count * 0.1)
        
        return round(min(1.0, score), 2)

    def check_process_supervision(self, reasoning: str) -> float:
        """
        [PAPER TRACEABILITY: Process Supervision (Lightweight PRM)]
        E. PROCESS SUPERVISION (STEP-AWARE REWARD)
        Validates reasoning steps (basic heuristics).
        Penalizes logical jumps and rewards structured step-by-step reasoning.
        """
        lower_r = reasoning.lower()
        score = 0.0
        
        # Check stepwise structure
        if "step 1" in lower_r and "step 2" in lower_r:
            score += 0.5
        elif "first" in lower_r and ("then" in lower_r or "next" in lower_r):
            score += 0.3
            
        # Penalize missing steps if it's very short but claims complex operations
        if len(lower_r.split()) < 10 and ("=" in lower_r or "so" in lower_r):
            score -= 0.5 # Logical jump penalty
            
        return max(-1.0, min(1.0, score))

    def check_reflection(self, reasoning: str, c: float) -> float:
        """
        [PAPER TRACEABILITY: Reflection Module]
        H. REFLECTION MODULE
        Model generates "What could be wrong?"
        Penalize if contradiction with final answer, reward correct self-correction.
        """
        lower_r = reasoning.lower()
        score = 0.0
        
        reflection_phrases = ["what could be wrong", "wait,", "let me check", "alternatively"]
        if any(phrase in lower_r for phrase in reflection_phrases):
            # Reflection attempted
            if c >= 1.0:
                score += 1.0 # Correct self-correction / successful verification
            else:
                score -= 0.5 # Contradiction or failed correction
                
        return score

    def check_numerical_integration(self, prediction: str, sympy_f: Any) -> bool:
        """
        [PAPER TRACEABILITY: Section 3.1.3 Solution Verification]
        Numerical multi-point quadrature verification.
        Instead of evaluating integrals, we differentiate the prediction F_pred(x)
        and compare it to the ground truth integrand f(x) at 5 random points.
        """
        import sympy as sp
        import random
        x = sp.Symbol('x')
        try:
            # Clean prediction string
            clean_pred = prediction.strip()
            if "Answer:" in clean_pred:
                clean_pred = clean_pred.split("Answer:")[-1].strip()
            clean_pred = clean_pred.replace("+ C", "").replace("+C", "").strip()
            
            F_pred = sp.parse_expr(clean_pred)
            f_pred = sp.diff(F_pred, x)
            
            # Evaluate at 5 random points
            for _ in range(5):
                test_point = random.uniform(-5, 5)
                p_val = float(f_pred.subs(x, test_point).evalf())
                t_val = float(sympy_f.subs(x, test_point).evalf())
                
                # Paper uses 10^-2 relative tolerance
                if not math.isclose(p_val, t_val, rel_tol=1e-2, abs_tol=1e-2):
                    return False
            return True
        except Exception:
            return False

    def verify(self, reasoning: str, prediction: str, ground_truth: str, sympy_f: Any = None) -> Tuple[float, float, float, float]:
        """
        Run all verifiers. 
        Returns Correctness (C), Reasoning Quality (Q), Process Supervision (P), and Reflection (R).
        """
        c = 0.0
        if self.check_exact_match(prediction, ground_truth):
            c = 1.0
        elif sympy_f is not None and self.check_numerical_integration(prediction, sympy_f):
            c = 1.0
        elif self.check_numeric_tolerance(prediction, ground_truth):
            c = 1.0
        elif self.check_python_execution(prediction, ground_truth):
            c = 1.0
            
        q = self.mock_llm_judge(reasoning, prediction, ground_truth)
        
        p = self.check_process_supervision(reasoning)
        r = self.check_reflection(reasoning, c)
        
        return c, q, p, r

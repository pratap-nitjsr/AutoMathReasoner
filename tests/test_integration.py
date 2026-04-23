import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import AutomathreasonerEnvironment
from env.models import AutomathreasonerAction

def test_integration_flow():
    env = AutomathreasonerEnvironment()
    obs = env.reset()
    
    print(f"PROBLEM: {obs.problem_text}")
    print(f"TRUE SOLUTION (CLEAN): {env.current_solution}")
    
    # 1. Correct Answer Test
    action = AutomathreasonerAction(
        reasoning="Integrating term by term...",
        final_answer=env.current_solution.replace(" + C", "")
    )
    step_obs = env.step(action)
    print(f"CORRECT ANSWER REWARD: {step_obs.reward}")
    print(f"METADATA: {step_obs.metadata}")
    
    assert step_obs.metadata['is_correct'] == True
    
    # 2. Wrong Answer Test
    env.reset()
    action_wrong = AutomathreasonerAction(
        reasoning="Bad math...",
        final_answer="x^99"
    )
    step_obs_wrong = env.step(action_wrong)
    print(f"WRONG ANSWER REWARD: {step_obs_wrong.reward}")
    assert step_obs_wrong.metadata['is_correct'] == False

if __name__ == "__main__":
    test_integration_flow()

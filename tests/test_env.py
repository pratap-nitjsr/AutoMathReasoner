import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.generator import TaskGenerationEngine
from env.verifier import VerifierSystem
from env.rewards import RewardSystem
from env.environment import AutomathreasonerEnvironment
from env.models import AutomathreasonerAction

def test_generator():
    engine = TaskGenerationEngine()
    
    # Test arithmetic
    prob, diff, ans = engine.generate_arithmetic(complexity=1)
    assert prob and ans
    
    # Test overall generate task
    task = engine.generate_task(target_difficulty_band=2.0)
    assert "problem" in task
    assert "solution" in task
    assert "difficulty" in task

def test_verifier():
    verifier = VerifierSystem()
    
    # Exact match
    assert verifier.check_exact_match("42", "42")
    assert verifier.check_exact_match(" 42 ", "42")
    
    # Numeric tolerance
    assert verifier.check_numeric_tolerance("3.14159", "3.1415")
    assert not verifier.check_numeric_tolerance("4.1415", "3.1415")
    
    # Python execution
    assert verifier.check_python_execution("2 + 2", "4")
    
    # Full verification
    c, q = verifier.verify("Because 2 + 2 is 4", "4", "4")
    assert c == 1.0
    assert q > 0.0  # Should have some mock reasoning score

def test_rewards():
    reward_sys = RewardSystem(max_len=1000)
    history = [{"final_answer": "42"}]
    
    # Test diversity drop on repeat
    d = reward_sys.compute_diversity("42", history)
    assert d == -1.0
    
    # Normal compute
    r, comps = reward_sys.compute_reward(
        correctness=1.0, 
        reasoning_quality=1.0, 
        action_str="step 1: do math. = 42", 
        final_answer="42",
        history=[], 
        times_seen_problem=0
    )
    assert r > 0.0

def test_environment_step():
    env = AutomathreasonerEnvironment()
    obs = env.reset()
    
    assert obs.problem_text != ""
    assert obs.difficulty_level > 0
    assert len(obs.history) == 0
    
    # Create action where they just pass dummy stuff
    action = AutomathreasonerAction(
        reasoning="I am guessing the answer.",
        final_answer="0"
    )
    
    obs_after = env.step(action)
    assert obs_after.reward is not None
    assert len(obs_after.history) == 1
    assert "reward_components" in obs_after.metadata

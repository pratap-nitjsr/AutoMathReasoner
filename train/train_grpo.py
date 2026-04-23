import random
import collections
import torch
import numpy as np
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import AutomathreasonerEnvironment
from env.models import AutomathreasonerAction

class ReplayBuffer:
    def __init__(self):
        self.ladder_buffer = []  # A. LADDER-STYLE self-bootstrapping buffer
        self.failed = []         # F. HARD NEGATIVE MINING buffer
        self.all_history = []
        
    def add_ladder(self, item):
        """
        [PAPER TRACEABILITY: LADDER-Style Self-Bootstrapping]
        Stores only high-quality trajectories.
        """
        self.ladder_buffer.append(item)
        # Keep top 20% effectively by hard capping and sorting if applicable
        # Simplistic version: Just keep recent highest
        if len(self.ladder_buffer) > 200:
            self.ladder_buffer.sort(key=lambda x: x['reward'], reverse=True)
            self.ladder_buffer = self.ladder_buffer[:100]

    def add(self, problem, best_solution, failed_attempts, reward=0.0):
        item = {
            "prompt": problem,
            "best_solution": best_solution,
            "failed_attempts": failed_attempts,
            "reward": reward
        }
        self.all_history.append(item)
        
        # F. HARD NEGATIVE MINING
        # Prioritize tracking failed problems
        if failed_attempts:
            # We explicitly track failures to reintroduce them
            self.failed.append(item)
            if len(self.failed) > 200:
                self.failed.pop(0)

    def sample(self, batch_size) -> list:
        """
        [PAPER TRACEABILITY: Hard Negative Mining]
        Samples from Ladder/High-quality, Failed, and Random.
        """
        if len(self.all_history) < batch_size:
            return self.all_history
            
        n_ladder = int(batch_size * 0.5)
        n_failed = int(batch_size * 0.3)
        n_random = batch_size - n_ladder - n_failed
        
        batch = []
        batch.extend(random.choices(self.ladder_buffer if self.ladder_buffer else self.all_history, k=n_ladder))
        batch.extend(random.choices(self.failed if self.failed else self.all_history, k=n_failed))
        batch.extend(random.choices(self.all_history, k=n_random))
        
        return batch

def main():
    max_seq_length = 1024
    # Load model via Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "llama-3-8b-instruct", 
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    
    env = AutomathreasonerEnvironment()
    replay_buffer = ReplayBuffer()
    
    # Generate some initial experiences
    initial_prompts = []
    for _ in range(50):
        obs = env.reset()
        initial_prompts.append({"prompt": obs.problem_text})
        
    dataset = Dataset.from_list(initial_prompts)
    
    def compute_rewards(prompts, completions, **kwargs):
        """
        [PAPER TRACEABILITY: GRPO (Group-Relative Policy Optimization)]
        D. GROUP-RELATIVE TRAINING
        TRL GRPOTrainer automatically handles the relative optimization aspect:
        log π(best) − log π(worst) by using the normalized rewards returned here.
        """
        rewards = []
        
        # C. SELF-CONSISTENCY SAMPLING
        # We group generated outputs by prompt to find the majority answer
        # TRL provides completions aligned with prompts. Usually completions are batched by K per prompt.
        prompt_answers = collections.defaultdict(list)
        
        parsed_actions = []
        for prompt, completion in zip(prompts, completions):
            try:
                parts = completion.split("Answer:")
                reasoning = parts[0].strip()
                answer = parts[1].strip() if len(parts) > 1 else ""
            except Exception:
                reasoning = completion
                answer = ""
                
            parsed_actions.append((prompt, completion, reasoning, answer))
            prompt_answers[prompt].append(answer)
            
        majority_answers = {}
        for p, ans_list in prompt_answers.items():
            if ans_list:
                majority_answers[p] = collections.Counter(ans_list).most_common(1)[0][0]

        for p, c, r, a in parsed_actions:
            action = AutomathreasonerAction(reasoning=r, final_answer=a)
            
            # Simulate step 
            env.reset()
            env.current_problem = p
            step_obs = env.step(action)
            r_total = step_obs.reward
            
            # [PAPER TRACEABILITY: Self-Consistency Sampling]
            # Verify majority match
            majority = majority_answers.get(p, "")
            is_majority = (a == majority) and len(a) > 0
            if is_majority:
                r_total += 0.2  # Bonus reward for mapping to majority
                
            rewards.append(r_total)
            
            is_correct = step_obs.metadata.get('is_correct', False)
            q_score = step_obs.metadata.get('reward_components', {}).get('Q_reasoning', 0.0)
            
            # B. ReST-STYLE FILTERING (SELF-TRAINING)
            # Filter samples where correctness = 1 AND reasoning quality > 0.6
            # [PAPER TRACEABILITY: ReST (Rest-Style Filtering)]
            if is_correct and q_score > 0.6:
                # Store as High Quality trajectory in Ladder buffer
                ladder_item = {
                    "prompt": p,
                    "best_solution": c,
                    "failed_attempts": [],
                    "reward": r_total
                }
                replay_buffer.add_ladder(ladder_item)

            # Standard buffer mapping
            if is_correct:
                replay_buffer.add(p, c, [], reward=r_total)
            else:
                replay_buffer.add(p, "", [c], reward=r_total)
                
        return rewards

    training_args = GRPOConfig(
        output_dir="outputs",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_prompt_length=128,
        max_completion_length=256,
        num_generations=8, # K=8 outputs per problem (Allows Self-consistency majority to work)
        max_steps=100,
        logging_steps=10,
    )
    
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[compute_rewards],
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting GRPO Training with Research-Aligned Modules...")
    trainer.train()

if __name__ == "__main__":
    main()

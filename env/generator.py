import random
from typing import Dict, Any, Tuple

class TaskGenerationEngine:
    def __init__(self):
        # Templates for different types of problems
        self.arithmetic_templates = [
            "What is {a} + {b}?",
            "Calculate {a} - {b}.",
            "Find the product of {a} and {b}.",
            "What is {a} divided by {b}?"
        ]
        self.algebra_templates = [
            "Solve for x: {a}x + {b} = {c}",
            "If {a}y - {b} = {c}, what is y?"
        ]
        self.word_problem_templates = [
            "John has {a} apples. He buys {b} more. Then he gives away {c}. How many apples does John have now?",
            "A train travels at {a} km/h for {b} hours. How far does it travel?"
        ]

    def _score_difficulty(self, steps: int, complexity: int, operations: int) -> float:
        """
        D = steps_required + number_complexity + operations_count
        """
        return float(steps + complexity + operations)

    def generate_arithmetic(self, complexity: int) -> Tuple[str, float, str]:
        a = random.randint(1 * complexity, 10 * complexity)
        b = random.randint(1 * complexity, 10 * complexity)
        op = random.choice(['+', '-', '*', '/'])
        
        operations = 1
        steps = 1
        
        if op == '+':
            problem = f"What is {a} + {b}?"
            answer = str(a + b)
        elif op == '-':
            problem = f"Calculate {a} - {b}."
            answer = str(a - b)
        elif op == '*':
            problem = f"Find the product of {a} and {b}."
            answer = str(a * b)
        elif op == '/':
            # Ensure divisible
            b = max(1, b)
            a = a * b
            problem = f"What is {a} divided by {b}?"
            answer = str(a // b)
            
        difficulty = self._score_difficulty(steps, complexity, operations)
        return problem, difficulty, answer

    def generate_algebra(self, complexity: int) -> Tuple[str, float, str]:
        a = random.randint(1, 5 * complexity)
        x = random.randint(1, 10)
        b = random.randint(1, 10 * complexity)
        op = random.choice(['+', '-'])
        
        operations = 2
        steps = 2
        
        if op == '+':
            c = a * x + b
            problem = f"Solve for x: {a}x + {b} = {c}"
        else:
            c = a * x - b
            problem = f"If {a}x - {b} = {c}, what is x?"
            
        answer = str(x)
        difficulty = self._score_difficulty(steps, complexity, operations)
        return problem, difficulty, answer

    def generate_word_problem(self, complexity: int) -> Tuple[str, float, str]:
        t = random.choice([0, 1])
        operations = 2
        steps = 2
        
        if t == 0:
            a = random.randint(5 * complexity, 15 * complexity)
            b = random.randint(2 * complexity, 10 * complexity)
            c = random.randint(1, a + b)
            problem = f"John has {a} apples. He buys {b} more. Then he gives away {c}. How many apples does John have now?"
            answer = str(a + b - c)
        else:
            a = random.randint(20 * complexity, 60 * complexity)
            b = random.randint(1, 5 * complexity)
            problem = f"A train travels at {a} km/h for {b} hours. How far does it travel?"
            answer = str(a * b)
            operations = 1
            steps = 1
            
        difficulty = self._score_difficulty(steps, complexity, operations)
        return problem, difficulty, answer

    def generate_task(self, target_difficulty_band: float) -> Dict[str, Any]:
        """
        Generate a task targeting a general difficulty band.
        target_difficulty_band can guide the complexity parameter.
        """
        complexity = max(1, int(target_difficulty_band / 2))
        
        prob_type = random.choices(
            ['arithmetic', 'algebra', 'word_problem'], 
            weights=[1, max(0.5, complexity-1), max(0.5, complexity-1)]
        )[0]
        
        if prob_type == 'arithmetic':
            problem, diff, ans = self.generate_arithmetic(complexity)
        elif prob_type == 'algebra':
            problem, diff, ans = self.generate_algebra(complexity)
        else:
            problem, diff, ans = self.generate_word_problem(complexity)
            
        return {
            "problem": problem,
            "difficulty": diff,
            "solution": ans,
            "type": prob_type
        }

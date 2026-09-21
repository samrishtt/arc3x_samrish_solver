"""
ECC Agent Evaluator Agent
Meta-evaluator monitoring trajectory quality, convergence rates, and win conditions.
"""

class AgentEvaluator:
    """Meta-evaluates trajectory health and goal convergence."""
    
    def __init__(self):
        self.role = "Agent Evaluator"

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC Agent-Evaluator]: Meta-evaluate progress toward the global victory condition:\n"
            "  - Evaluate progress metrics: target distance reduction, key acquisition, obstacles cleared.\n"
            "  - If after 5 actions no measurable progress is achieved, trigger an immediate hypothesis pivot.\n"
            "  - Benchmark the current state against the initial frame to ensure net-positive movement."
        )

"""
ECC Planner Agent
Specialized in hierarchical planning, goal decomposition, and step-budget optimization.
"""
from typing import Dict, Any, List, Optional

class Planner:
    """Formulates multi-step trajectories and sub-goal hierarchies."""
    
    def __init__(self, name: str = "ecc:planner"):
        self.name = name

    def plan_trajectory(self, start: Any, goal: Any, budget: int = 100) -> List[Any]:
        """Generates candidate plan under given step budget constraints."""
        return []

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC Planner]: Formulate hierarchical action sequences:\n"
            "  - Decompose the global victory condition into discrete sub-goals.\n"
            "  - Budget actions judiciously; avoid speculative single-step wandering.\n"
            "  - Prioritize shortest path trajectories to active sub-goals.\n"
            "  - When a plan path is clear, emit macro action batches `action([...])`."
        )

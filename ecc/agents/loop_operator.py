"""
ECC Loop Operator Agent
Specialized in autonomous loop execution, stall monitoring, and intervention protocols.
"""
from typing import Dict, Any, List, Optional

class LoopOperator:
    """Monitors the execution loop and ensures continuous progress."""
    
    def __init__(self, name: str = "ecc:loop-operator"):
        self.name = name

    def check_progress(self, current_step: int, last_score_change_step: int, threshold: int = 20) -> bool:
        """Determines if the agent is stuck or stalling."""
        return (current_step - last_score_change_step) < threshold

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC Loop-Operator]: Maintain continuous forward momentum:\n"
            "  - Monitor step budget expenditure vs score accumulation.\n"
            "  - If stagnation occurs, execute an exploratory branch shift.\n"
            "  - Terminate or pivot unproductive hypotheses swiftly."
        )

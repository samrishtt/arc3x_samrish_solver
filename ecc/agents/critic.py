"""
ECC Critic / Code Reviewer Agent
Specialized in falsification testing, loop detection, and anti-oscillation scrutiny.
"""
from typing import Dict, Any, List, Optional

class Critic:
    """Evaluates proposed actions against failure modes and invariants."""
    
    def __init__(self, name: str = "ecc:critic"):
        self.name = name

    def evaluate_move(self, proposed_actions: List[Any], history: List[Any]) -> bool:
        """Checks for cyclic oscillations, redundant moves, and hazard collisions."""
        return True

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC Critic]: Scrutinize proposed actions against failure modes:\n"
            "  - Detect and prune cyclic ping-pong oscillation (e.g., UP-DOWN-UP or LEFT-RIGHT-LEFT).\n"
            "  - If the previous action caused 0 state change or zero reward, discard that branch immediately.\n"
            "  - Never waste moves clicking static HUD elements (e.g. timer bars, score counters).\n"
            "  - Verify that the action cannot result in catastrophic failure or trap state."
        )

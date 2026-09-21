"""
ECC Silent Failure Hunter Agent
Identifies subtle failure modes, zero-delta transitions, invisible barriers, and unrewarded exploration.
"""

class SilentFailureHunter:
    """Hunts down silent non-progression, zero-delta states, and unhandled traps."""
    
    def __init__(self):
        self.role = "Silent Failure Hunter"

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC Silent-Failure-Hunter]: Detect and eliminate zero-delta or deceptive moves:\n"
            "  - Inspect `last_action_result`: if `board_changed` is False, the action failed silently (e.g. wall collision).\n"
            "  - Never repeat an action that produced no state transition without a clear causal justification.\n"
            "  - Detect trap states: if an object cannot reach any valid target, backtrack or reset immediately.\n"
            "  - Ensure every executed move produces verifiable entropy reduction in the puzzle space."
        )

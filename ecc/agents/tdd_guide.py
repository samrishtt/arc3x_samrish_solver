"""
ECC TDD Guide Agent
Applies Test-Driven Deliberation: states pre-conditions and expected post-conditions before acting.
"""

class TDDGuide:
    """Enforces test-driven mental simulation and pre/post assertions."""
    
    def __init__(self):
        self.role = "TDD Guide"

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC TDD-Guide]: Apply Test-Driven Reasoning before any action execution:\n"
            "  - Pre-condition: State what must be true about `current_frame` before acting.\n"
            "  - Post-condition: State what MUST be true in the resulting frame if the hypothesis is correct.\n"
            "  - Verify assertion: Check if post-action result matches expectations; if not, reject hypothesis."
        )

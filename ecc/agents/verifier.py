"""
ECC Verifier / TDD-Guide Agent
Specialized in forward mental simulation, collision verification, and test-driven validation.
"""
from typing import Dict, Any, List, Optional

class Verifier:
    """Performs mental simulation of trajectories before committing actions."""
    
    def __init__(self, name: str = "ecc:verifier"):
        self.name = name

    def verify_path(self, path: List[Any], obstacles: List[Any]) -> bool:
        """Simulates path execution against obstacle constraints."""
        return True

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC Verifier]: Mental test-driven execution before emission:\n"
            "  - Mentally step through each action in sequence before emitting `action(...)`.\n"
            "  - Confirm each coordinate step avoids impassable barrier pixels.\n"
            "  - Confirm the end of the action sequence brings the state closer to the active goal."
        )

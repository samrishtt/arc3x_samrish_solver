"""
ECC Spec Miner / Reverse Engineer Agent
Extracts the underlying formal grammar, physics laws, and transition dynamics of the environment.
"""

class SpecMiner:
    """Reverse engineers rules, object interactions, and transition invariants."""
    
    def __init__(self):
        self.role = "Spec Miner"

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC Spec-Miner]: Reverse-engineer the hidden physics and game mechanics:\n"
            "  - Formulate hypotheses on object roles: Avatar, Key, Door, Blocker, Goal, Hazard.\n"
            "  - Document interaction rules: 'Contact with color C triggers event E'.\n"
            "  - Track cross-level rule transfer: carry forward confirmed mechanics while verifying new variations.\n"
            "  - State explicit transition axioms before executing multi-step sequences."
        )

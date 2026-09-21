"""
ECC Anti-Oscillation Rule
Detects and prunes repetitive cyclical action loops.
"""

class AntiOscillationRule:
    """Prevents cyclical ping-pong traps and redundant state oscillation."""
    
    @staticmethod
    def get_prompt_rule() -> str:
        return (
            "- Anti-Oscillation: Never oscillate between alternating actions (e.g. UP-DOWN-UP or LEFT-RIGHT-LEFT) "
            "when the board state does not change. Cycles indicate failed hypotheses and must be broken immediately."
        )

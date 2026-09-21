"""
ECC Anti-HUD Rule Engine
Protects the agent from wasting budget on timers, scoreboards, and static border artifacts.
"""
from typing import Tuple, Optional

class AntiHUDRule:
    """Provides heuristic boundaries to separate playable arena from UI overlays."""

    @staticmethod
    def is_hud_region(y: int, x: int, height: int, width: int) -> bool:
        """Determines if a coordinate is within top or bottom HUD bands."""
        if y < 3 or y >= (height - 3):
            return True
        return False

    @staticmethod
    def get_prompt_rule() -> str:
        return (
            "- Anti-HUD Rule: Disregard decorative borders, score numbers, and decrementing timer bars. "
            "Never target actions at non-interactive HUD elements."
        )

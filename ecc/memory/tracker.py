"""
ECC Action Tracker
Tracks trajectory history and detects oscillation patterns.
"""
from typing import List, Any

class ActionTracker:
    """Monitors historical actions to detect loops or stalls."""
    
    def __init__(self, max_history: int = 50):
        self.history: List[Any] = []
        self.max_history = max_history

    def push(self, action: Any) -> None:
        self.history.append(action)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def is_oscillating(self) -> bool:
        """Detects ping-pong patterns like A-B-A-B."""
        if len(self.history) < 4:
            return False
        return self.history[-1] == self.history[-3] and self.history[-2] == self.history[-4]

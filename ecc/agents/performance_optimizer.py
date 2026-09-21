"""
ECC Performance Optimizer Agent
Maximizes step-budget efficiency, optimizes macro-action batching, and prunes wasteful traversal.
"""

class PerformanceOptimizer:
    """Optimizes trajectory efficiency and action-sequence batching."""
    
    def __init__(self):
        self.role = "Performance Optimizer"

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC Performance-Optimizer]: Maximize action budget efficiency and speed:\n"
            "  - Group validated paths into batch execution calls: `action(['UP', 'UP', 'RIGHT'])`.\n"
            "  - Avoid turn-by-turn micro-stepping once a path or sequence is mathematically verified.\n"
            "  - Minimize total actions per level to preserve remaining step budget for subsequent puzzles.\n"
            "  - Prune redundant moves, loops, and speculative dead-end exploration."
        )

"""
ECC Harness Optimizer Agent
Coordinates multi-worker execution, memory vault synchronization, and concurrency management.
"""

class HarnessOptimizer:
    """Optimizes concurrency, memory vault synchronization, and runtime execution."""
    
    def __init__(self):
        self.role = "Harness Optimizer"

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC Harness-Optimizer]: Synchronize runtime state and concurrency:\n"
            "  - Maintain global memory vault state across turns.\n"
            "  - Prevent thread starvation and memory leaks during multi-game concurrency."
        )

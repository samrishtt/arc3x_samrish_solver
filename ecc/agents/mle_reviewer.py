"""
ECC MLE Reviewer Agent
Analyzes visual distribution, spatial representations, and entropy in grid states.
"""

class MLEReviewer:
    """Reviews state representations and visual grid dynamics."""
    
    def __init__(self):
        self.role = "MLE Reviewer"

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC MLE-Reviewer]: Analyze spatial feature representations and state entropy:\n"
            "  - Treat grid segmentation as a categorical graph of connected visual primitives.\n"
            "  - Identify minimal energy configurations that satisfy puzzle symmetry and balance."
        )

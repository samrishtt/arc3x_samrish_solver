"""
ECC Architect Agent
Specialized in system design, spatial scene analysis, and multi-modal scene decomposition.
"""
from typing import Dict, Any, List, Optional

class Architect:
    """Decomposes the environment state into structural and semantic components."""
    
    def __init__(self, name: str = "ecc:architect"):
        self.name = name

    def analyze_frame(self, frame_data: Any) -> Dict[str, Any]:
        """Analyzes spatial topology, identifies player coordinate, goals, and static barriers."""
        analysis = {
            "agent": self.name,
            "spatial_topology": "2D Grid / Discrete Coordinates",
            "detected_objects": [],
            "hypotheses": []
        }
        return analysis

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC Architect]: Deconstruct the visual grid into functional entities:\n"
            "  - Identify the player avatar/agent position.\n"
            "  - Identify interactive objects, targets, keys, and doors.\n"
            "  - Map impassable obstacles and hazards.\n"
            "  - Maintain spatial invariance under D4 transformations (rotations/reflections)."
        )

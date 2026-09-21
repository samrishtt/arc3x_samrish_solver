"""
ECC Human Priors Rule Engine
Defines fundamental Core Knowledge priors for ARC reasoning.
"""
from typing import List, Dict, Any

class HumanPriors:
    """Encodes objectness, spatial invariance, goal-directed agency, and basic physics."""
    
    PRIORS: List[str] = [
        "Objectness: Cohesive, connected color components are bounded persistent physical objects.",
        "Goal-Directed Agency: The agent moves purposefully to interact with key objects or destinations.",
        "Spatial Symmetry: Invariance under dihedral group D4 transformations (rotation, reflection).",
        "Topological Connectivity: Trajectories require continuous contiguous paths without jumping barriers."
    ]

    @classmethod
    def get_summary(cls) -> str:
        return "\n".join(f"- {p}" for p in cls.PRIORS)

"""
ECC D4 Invariance Rule
Enforces Dihedral Group D4 geometric invariance (rotations and reflections).
"""

class D4InvarianceRule:
    """Enforces spatial invariance across all 8 dihedral transformations."""
    
    @staticmethod
    def get_prompt_rule() -> str:
        return (
            "- D4 Invariance: Grid puzzles often exhibit 8-fold dihedral symmetry (rotations of 0, 90, 180, 270 degrees "
            "and horizontal/vertical/diagonal reflections). Test whether the solution rule is invariant under D4 transforms."
        )

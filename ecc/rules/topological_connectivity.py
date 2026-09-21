"""
ECC Topological Connectivity Rule
Enforces path existence and flood-fill connectivity priors.
"""

class TopologicalConnectivityRule:
    """Enforces continuous path connectivity and obstacle avoidance."""
    
    @staticmethod
    def get_prompt_rule() -> str:
        return (
            "- Topological Connectivity: Physical navigation requires a 4-connected continuous path of non-barrier cells. "
            "Use BFS or flood-fill in Python to verify path existence before committing to a navigation sequence."
        )

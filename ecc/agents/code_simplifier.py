"""
ECC Code Simplifier Agent
Maintains ultra-clean, sandboxed Python code snippets and compact decision-oriented outputs.
"""

class CodeSimplifier:
    """Ensures tool-call Python code is robust, minimal, and error-free."""
    
    def __init__(self):
        self.role = "Code Simplifier"

    def get_prompt_instruction(self) -> str:
        return (
            "[ECC Code-Simplifier]: Write clean, robust, decision-oriented Python snippets:\n"
            "  - Avoid massive loops or excessive printouts; return compact summary statistics and diffs.\n"
            "  - Write self-contained algorithms (BFS, DFS, flood-fill) using only standard library tools.\n"
            "  - Handle boundary checks defensively to avoid IndexError or NoneType exceptions in the sandbox."
        )

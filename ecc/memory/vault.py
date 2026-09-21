"""
ECC Memory Vault
Maintains persistent cross-step memory, epistemic ledger, and hypothesis states.
"""
from typing import Dict, Any, List, Set

class MemoryVault:
    """Stores verified environmental facts, visited locations, and falsified rules."""
    
    def __init__(self):
        self.verified_facts: Dict[str, Any] = {}
        self.falsified_hypotheses: Set[str] = set()
        self.visited_coordinates: Set[tuple] = set()
        self.object_inventory: Dict[str, Any] = {}

    def record_falsification(self, hypothesis: str) -> None:
        self.falsified_hypotheses.add(hypothesis)

    def record_fact(self, key: str, value: Any) -> None:
        self.verified_facts[key] = value

    def is_falsified(self, hypothesis: str) -> bool:
        return hypothesis in self.falsified_hypotheses

    def reset(self) -> None:
        self.verified_facts.clear()
        self.falsified_hypotheses.clear()
        self.visited_coordinates.clear()
        self.object_inventory.clear()

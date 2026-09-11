"""
Memory subsystem for World Model Lab.

Provides:
- 4-layer hierarchical memory taxonomy:
  * RunMemory (One run)
  * GameMemory (Across versions)
  * GlobalMemory (Across entire project)
  * SubmissionMemory (Frozen at competition submission)
- HierarchicalMemoryManager
- EpisodicSemanticMemory
- PersistentMemoryBank
"""

from world_model_lab.memory.store import EpisodicSemanticMemory, EpisodicRecord
from world_model_lab.memory.persistent_bank import PersistentMemoryBank, CausalRuleRecord, RunMetadata
from world_model_lab.memory.hierarchical_memory import (
    RunMemory,
    GameMemory,
    GlobalMemory,
    SubmissionMemory,
    HierarchicalMemoryManager,
    DistilledRunSummary,
    DistilledRule
)

__all__ = [
    "EpisodicSemanticMemory",
    "EpisodicRecord",
    "PersistentMemoryBank",
    "CausalRuleRecord",
    "RunMetadata",
    "RunMemory",
    "GameMemory",
    "GlobalMemory",
    "SubmissionMemory",
    "HierarchicalMemoryManager",
    "DistilledRunSummary",
    "DistilledRule"
]

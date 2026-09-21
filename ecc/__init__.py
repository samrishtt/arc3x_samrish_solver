"""
ECC (Everything Claude Code) - Full Multi-Agent Swarm for ARC-AGI-3
"""
from .harness import ECCHarness, get_harness, attach_to_solver
from . import agents
from . import memory
from . import rules

__version__ = "2.2.0-full-fleet"
__all__ = ["ECCHarness", "get_harness", "attach_to_solver", "agents", "memory", "rules"]

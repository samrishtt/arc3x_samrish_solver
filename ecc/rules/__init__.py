"""
ECC Rules Package
Exports domain-specific cognitive priors and invariance rules.
"""
from .anti_hud import AntiHUDRule
from .human_priors import HumanPriors
from .d4_invariance import D4InvarianceRule
from .anti_oscillation import AntiOscillationRule
from .topological_connectivity import TopologicalConnectivityRule

__all__ = [
    "AntiHUDRule",
    "HumanPriors",
    "D4InvarianceRule",
    "AntiOscillationRule",
    "TopologicalConnectivityRule"
]

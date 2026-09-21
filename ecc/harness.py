"""
ECC Agent Harness for ARC-AGI-3 (Full Everything Claude Code Fleet)
Integrates the complete 13-agent cognitive architecture, memory vault, and human priors into the solver.
"""
from typing import Optional, Any
from .agents import (
    Architect, Planner, Critic, Verifier, LoopOperator,
    SilentFailureHunter, PerformanceOptimizer, SpecMiner,
    CodeSimplifier, AgentEvaluator, TDDGuide, HarnessOptimizer, MLEReviewer
)
from .memory import MemoryVault, ActionTracker, SpatialCache
from .rules import (
    HumanPriors, AntiHUDRule, D4InvarianceRule,
    AntiOscillationRule, TopologicalConnectivityRule
)

class ECCHarness:
    """The central orchestrator connecting the full ECC multi-agent swarm to the model runtime."""
    
    def __init__(self):
        self.architect = Architect()
        self.planner = Planner()
        self.critic = Critic()
        self.verifier = Verifier()
        self.loop_operator = LoopOperator()
        self.silent_failure_hunter = SilentFailureHunter()
        self.performance_optimizer = PerformanceOptimizer()
        self.spec_miner = SpecMiner()
        self.code_simplifier = CodeSimplifier()
        self.agent_evaluator = AgentEvaluator()
        self.tdd_guide = TDDGuide()
        self.harness_optimizer = HarnessOptimizer()
        self.mle_reviewer = MLEReviewer()
        
        self.memory = MemoryVault()
        self.tracker = ActionTracker()
        self.spatial_cache = SpatialCache()

    def generate_cognitive_protocol(self) -> str:
        """Constructs the full multi-agent prompt protocol containing all 13 specialized ECC agents."""
        protocol = [
            "\n=== ECC Complete Multi-Agent Swarm Deliberation Protocol ===",
            "Before generating actions with `action(...)`, deliberate using the full ECC agent fleet:\n",
            self.architect.get_prompt_instruction(),
            self.planner.get_prompt_instruction(),
            self.spec_miner.get_prompt_instruction(),
            self.tdd_guide.get_prompt_instruction(),
            self.verifier.get_prompt_instruction(),
            self.critic.get_prompt_instruction(),
            self.silent_failure_hunter.get_prompt_instruction(),
            self.performance_optimizer.get_prompt_instruction(),
            self.code_simplifier.get_prompt_instruction(),
            self.agent_evaluator.get_prompt_instruction(),
            self.loop_operator.get_prompt_instruction(),
            "\nCore Human Priors & Physical Invariants:",
            HumanPriors.get_summary(),
            AntiHUDRule.get_prompt_rule(),
            D4InvarianceRule.get_prompt_rule(),
            AntiOscillationRule.get_prompt_rule(),
            TopologicalConnectivityRule.get_prompt_rule(),
            "=== End ECC Complete Multi-Agent Protocol ===\n"
        ]
        return "\n".join(protocol)

    def attach(self, solver: Any, tool_agent: Optional[Any] = None) -> bool:
        """Cleanly attaches the full ECC cognitive protocol to the solver and tool agent."""
        attached = False
        protocol_str = self.generate_cognitive_protocol()

        if tool_agent is not None:
            if hasattr(tool_agent, "VISUAL_GAME_ADDENDUM"):
                tool_agent.VISUAL_GAME_ADDENDUM += protocol_str
                attached = True
            elif hasattr(tool_agent, "SYSTEM_PROMPT"):
                tool_agent.SYSTEM_PROMPT += protocol_str
                attached = True

        if solver is not None and hasattr(solver, "ecc_harness"):
            solver.ecc_harness = self

        return attached

_default_harness: Optional[ECCHarness] = None

def get_harness() -> ECCHarness:
    global _default_harness
    if _default_harness is None:
        _default_harness = ECCHarness()
    return _default_harness

def attach_to_solver(solver: Any, tool_agent: Optional[Any] = None) -> bool:
    """Convenience entrypoint to attach the full ECC swarm to the solver."""
    harness = get_harness()
    return harness.attach(solver, tool_agent)

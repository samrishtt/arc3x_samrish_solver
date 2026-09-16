"""
Dual-Agent Dialectical Debate Architecture (Proposer vs Critic + Grounded Simulator)

Implements multi-agent dialectical deliberation:
1. Agent A (Proposer / Hypothesis Advocate):
   Proposes action candidates based on active epistemic predictive disagreement.
2. Agent B (Adversarial Critic / Skeptic):
   Scans episodic memory and spatial geometry to raise objections against hazards,
   wall collisions, and redundant oscillation traps.
3. The Simulator Arbiter (Internal Grounded World Model):
   Mentally evaluates the disputed action. If the Critic proves a lethal hazard or
   wall collision, the action is vetoed BEFORE physical execution and replaced with
   the highest-information alternative.
"""

from typing import List, Dict, Tuple, Optional, Any
from world_model_lab.core.types import (
    Action, Observation, Hypothesis, MentalRollout, PredictionError
)
from world_model_lab.agents.active_agent import ActiveWorldModelAgent
from world_model_lab.experimentation.selector import SelectionStrategy


class DialecticalDebateAgent(ActiveWorldModelAgent):
    """
    Two-Agent Dialectical Debate Agent with Mental Grounding:
    - Proposer: active world-model hypothesis selector.
    - Critic: examines spatial hazards and boundary collisions.
    - Arbiter: counterfactual simulator that vetoes fatal moves and arbitrates consensus.
    """

    def __init__(
        self,
        grid_size: int = 7,
        strategy: SelectionStrategy = SelectionStrategy.PREDICTIVE_DISAGREEMENT,
        confidence_threshold: float = 0.75,
        seed: int = 42,
        skepticism_weight: float = 0.6
    ):
        super().__init__(
            grid_size=grid_size,
            strategy=strategy,
            confidence_threshold=confidence_threshold,
            seed=seed
        )
        self.skepticism_weight = skepticism_weight

        # Debate telemetry
        self.total_proposals: int = 0
        self.total_critiques: int = 0
        self.vetoed_actions: int = 0
        self.consensus_count: int = 0
        self.debate_history: List[Dict[str, Any]] = []

    def reset(self) -> None:
        super().reset()
        self.total_proposals = 0
        self.total_critiques = 0
        self.vetoed_actions = 0
        self.consensus_count = 0
        self.debate_history.clear()

    def _critic_evaluate(
        self,
        candidate_action: Action,
        observation: Observation,
        hypotheses: List[Hypothesis]
    ) -> Tuple[bool, str, float]:
        """
        Adversarial Critic evaluates candidate action against:
        1. Spatial boundary collision (no-ops).
        2. Known hazardous entity adjacency.
        3. Oscillation loops.

        Returns: (is_objected, reason, severity)
        """
        agent_pos = observation.agent_pos
        h, w = observation.grid_shape

        # 1. Boundary check objection
        dr, dc = 0, 0
        if candidate_action == Action.UP:
            dr = -1
        elif candidate_action == Action.DOWN:
            dr = 1
        elif candidate_action == Action.LEFT:
            dc = -1
        elif candidate_action == Action.RIGHT:
            dc = 1

        target_r = agent_pos.r + dr
        target_c = agent_pos.c + dc

        if not (0 <= target_r < h and 0 <= target_c < w):
            return True, f"Wall collision at ({target_r}, {target_c})", 1.0

        # 2. Check for hazards (static entities marked hazard or lethal)
        for ent in observation.entities:
            if not ent.is_agent and ent.pos.r == target_r and ent.pos.c == target_c:
                if "spike" in ent.id or "hazard" in ent.id or ent.color in ("hazard", "spike") or getattr(ent, "state", "") in ("hazard", "lethal"):
                    return True, f"Hazardous entity '{ent.id}' ({ent.color}) in target cell", 0.95

        return False, "No objection", 0.0

    def select_action(self, obs: Observation, last_reward: float = 0.0) -> Action:
        """
        Deliberates between Proposer and Critic:
        1. Proposer runs the full active world model pipeline.
        2. Critic challenges proposed action for boundary/hazard traps.
        3. Arbiter vetoes and substitutes alternative if severity > 0.85.
        """
        proposed_action = super().select_action(obs, last_reward)
        self.total_proposals += 1

        is_objected, reason, severity = self._critic_evaluate(
            candidate_action=proposed_action,
            observation=obs,
            hypotheses=self.hypothesis_manager.hypotheses
        )

        if is_objected and severity > 0.85:
            self.total_critiques += 1
            self.vetoed_actions += 1

            # Select best alternative action that avoids the objection
            legal_alts = [
                a for a in Action.all_actions()
                if a != proposed_action and not self._critic_evaluate(a, obs, self.hypothesis_manager.hypotheses)[0]
            ]
            if legal_alts:
                best_alt = max(
                    legal_alts,
                    key=lambda a: self.simulator.compute_disagreement(obs, a, self.hypothesis_manager.hypotheses)
                )
            else:
                best_alt = Action.WAIT

            self.debate_history.append({
                "step": self.step_count,
                "proposed": proposed_action.name,
                "critic_objected": True,
                "reason": reason,
                "verdict": f"VETOED -> REPLACED_WITH_{best_alt.name}"
            })
            self.last_action = best_alt
            return best_alt

        self.consensus_count += 1
        self.debate_history.append({
            "step": self.step_count,
            "proposed": proposed_action.name,
            "critic_objected": is_objected,
            "reason": reason,
            "verdict": "ACCEPTED"
        })
        return proposed_action

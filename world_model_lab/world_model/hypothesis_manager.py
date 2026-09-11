from typing import List, Dict, Optional, Tuple
import numpy as np
from world_model_lab.core.types import (
    Hypothesis, ConditionType, EffectType, Observation, Entity
)

class HypothesisManager:
    """
    All-Entity Hypothesis Manager:
    - Generates candidate explanatory hypotheses across ALL observable entities.
    - Represents uncertainty over the entire causal structure of the environment.
    - Does NOT require or accept any target entity identity, target color, or target state.
    - Maintains a normalized belief distribution P(H) across the full hypothesis space.
    """

    def __init__(self):
        self.hypotheses: List[Hypothesis] = []

    def reset(self) -> None:
        self.hypotheses.clear()

    def generate_candidate_hypotheses(self, obs: Observation) -> List[Hypothesis]:
        """
        Generates generic causal hypotheses over all observable non-agent entities:
        For every entity T (candidate target) and every cause C / pair (C1, C2):
        - H_touch: TOUCH(C) -> T:active
        - H_adj:   ADJACENT(C1, C2) -> T:active
        """
        self.hypotheses = []
        all_objects = [e for e in obs.entities if not e.is_agent]
        movable_objects = [e for e in all_objects if not e.is_static]

        colors = [e.color for e in all_objects]
        movable_colors = [e.color for e in movable_objects]

        # For every potential target entity in the environment
        for target_color in colors:
            # 1. Single-touch cause hypotheses
            for cause_color in colors:
                h = Hypothesis(
                    id=f"H_touch_{cause_color}_to_{target_color}",
                    condition=ConditionType.TOUCH,
                    cause_color=cause_color,
                    trigger_color=None,
                    effect=EffectType.STATE_CHANGE,
                    effect_target_color=target_color,
                    effect_new_state="active",
                    confidence=1.0
                )
                self.hypotheses.append(h)

            # 2. Pairwise interaction (adjacency) hypotheses
            for i in range(len(movable_colors)):
                for j in range(i + 1, len(colors)):
                    c1, c2 = sorted([movable_colors[i], colors[j]])
                    h = Hypothesis(
                        id=f"H_adj_{c1}_{c2}_to_{target_color}",
                        condition=ConditionType.ADJACENCY,
                        cause_color=c1,
                        trigger_color=c2,
                        effect=EffectType.STATE_CHANGE,
                        effect_target_color=target_color,
                        effect_new_state="active",
                        confidence=1.0
                    )
                    self.hypotheses.append(h)

        self._normalize_confidences()
        return self.hypotheses

    def get_best_hypothesis(self, target_color: Optional[str] = None) -> Optional[Hypothesis]:
        if not self.hypotheses:
            return None
        if target_color is not None:
            filtered = [h for h in self.hypotheses if h.effect_target_color == target_color]
            if filtered:
                return max(filtered, key=lambda h: h.confidence)
        return max(self.hypotheses, key=lambda h: h.confidence)

    def get_hypotheses_for_target(self, target_color: str) -> List[Hypothesis]:
        return [h for h in self.hypotheses if h.effect_target_color == target_color]

    def update_hypothesis(self, hypothesis_id: str, verified: bool, lr: float = 0.5) -> None:
        """
        Bayesian-style update:
        If verified: boost confidence.
        If falsified: penalize confidence sharply.
        """
        for h in self.hypotheses:
            if h.id == hypothesis_id:
                if verified:
                    h.success_count += 1
                    h.confidence *= (1.0 + lr)
                else:
                    h.falsification_count += 1
                    h.confidence *= (1.0 - lr)
                    if h.confidence < 1e-5:
                        h.confidence = 1e-5
        self._normalize_confidences()

    def _normalize_confidences(self) -> None:
        total = sum(h.confidence for h in self.hypotheses)
        if total > 0:
            for h in self.hypotheses:
                h.confidence /= total

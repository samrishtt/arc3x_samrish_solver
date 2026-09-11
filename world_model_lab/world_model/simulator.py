from typing import List, Dict, Tuple, Optional
import numpy as np
from world_model_lab.core.types import (
    Observation, Action, Position, Entity, Hypothesis,
    ConditionType, EffectType, MentalRollout
)

class CounterfactualSimulator:
    """
    All-Entity Internal Mental Simulator:
    - Rolls forward the agent's actions internally without executing them in reality.
    - Evaluates what each competing hypothesis predicts will occur across all entities.
    - Measures multi-entity predictive disagreement across the hypothesis set.
    """

    def __init__(self, grid_size: int = 7):
        self.grid_size = grid_size

    def simulate_forward(
        self,
        obs: Observation,
        action: Action,
        hypothesis: Hypothesis
    ) -> MentalRollout:
        """
        Simulates one step forward under the assumption that `hypothesis` is true.
        """
        predicted_events: List[str] = []
        # Clone entities for forward simulation
        sim_entities = [e.clone() for e in obs.entities]
        agent = next((e for e in sim_entities if e.is_agent), None)
        assert agent is not None

        interacted_entity: Optional[Entity] = None

        if action in Action.movement_actions():
            dr, dc = 0, 0
            if action == Action.UP:
                dr = -1
            elif action == Action.DOWN:
                dr = 1
            elif action == Action.LEFT:
                dc = -1
            elif action == Action.RIGHT:
                dc = 1

            new_r = np.clip(agent.pos.r + dr, 0, self.grid_size - 1)
            new_c = np.clip(agent.pos.c + dc, 0, self.grid_size - 1)
            target_pos = Position(int(new_r), int(new_c))

            blocker = next((e for e in sim_entities if not e.is_agent and e.pos.r == target_pos.r and e.pos.c == target_pos.c), None)
            if blocker:
                if not blocker.is_static:
                    push_r = np.clip(blocker.pos.r + dr, 0, self.grid_size - 1)
                    push_c = np.clip(blocker.pos.c + dc, 0, self.grid_size - 1)
                    push_pos = Position(int(push_r), int(push_c))
                    push_blocker = next((e for e in sim_entities if not e.is_agent and e.pos.r == push_pos.r and e.pos.c == push_pos.c), None)
                    if not push_blocker and (push_pos.r != blocker.pos.r or push_pos.c != blocker.pos.c):
                        blocker.pos = push_pos
                        agent.pos = target_pos
            else:
                agent.pos = target_pos

        elif action == Action.INTERACT:
            adj = [
                e for e in sim_entities
                if not e.is_agent and (agent.pos.is_adjacent(e.pos) or (agent.pos.r == e.pos.r and agent.pos.c == e.pos.c))
            ]
            if adj:
                interacted_entity = adj[0]

        # Test hypothesis condition on simulated state
        triggered = False
        if hypothesis.condition == ConditionType.TOUCH:
            if action == Action.INTERACT and interacted_entity:
                if interacted_entity.color == hypothesis.cause_color:
                    triggered = True

        elif hypothesis.condition == ConditionType.ADJACENCY:
            cause_ent = next((e for e in sim_entities if e.color == hypothesis.cause_color), None)
            trig_ent = next((e for e in sim_entities if e.color == hypothesis.trigger_color), None)
            if cause_ent and trig_ent:
                if cause_ent.pos.is_adjacent(trig_ent.pos):
                    triggered = True

        target_ent = next((e for e in sim_entities if e.color == hypothesis.effect_target_color), None)
        pred_state = "dormant"
        if target_ent:
            pred_state = target_ent.state

        if triggered:
            pred_state = hypothesis.effect_new_state
            predicted_events.append(f"STATE_CHANGE:{hypothesis.effect_target_color}:{pred_state}")

        return MentalRollout(
            action=action,
            hypothesis_id=hypothesis.id,
            predicted_events=predicted_events,
            predicted_target_state=pred_state,
            confidence=hypothesis.confidence
        )

    def compute_disagreement(
        self,
        obs: Observation,
        action: Action,
        hypotheses: List[Hypothesis]
    ) -> float:
        """
        Computes total predictive disagreement across all entities for candidate action A.
        Disagreement = sum_{target_T} Gini(P(state(T) | A))
        """
        if not hypotheses:
            return 0.0

        # Group hypotheses by target entity
        target_groups: Dict[str, List[Hypothesis]] = {}
        for h in hypotheses:
            target_groups.setdefault(h.effect_target_color, []).append(h)

        total_disagreement = 0.0

        for target_color, h_list in target_groups.items():
            weight_sum = sum(h.confidence for h in h_list)
            if weight_sum <= 0:
                continue

            state_prob_mass: Dict[str, float] = {}
            for h in h_list:
                rollout = self.simulate_forward(obs, action, h)
                norm_conf = h.confidence / weight_sum
                state_prob_mass[rollout.predicted_target_state] = (
                    state_prob_mass.get(rollout.predicted_target_state, 0.0) + norm_conf
                )

            # Gini impurity for this target
            target_disagreement = 1.0 - sum(p ** 2 for p in state_prob_mass.values())
            total_disagreement += max(0.0, float(target_disagreement))

        return float(total_disagreement)

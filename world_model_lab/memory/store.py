from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from world_model_lab.core.types import Observation, Action
from world_model_lab.perception.extractor import StateTransitionDiff, SpatialRelation

@dataclass
class EpisodicRecord:
    step: int
    action: Action
    prev_obs: Observation
    next_obs: Observation
    diff: StateTransitionDiff
    reward: float

class EpisodicSemanticMemory:
    """
    Episodic and Semantic Memory:
    - Episodic Memory: Chronological ledger of all experience transitions.
    - Semantic Memory: Statistical contingency table associating antecedents
      (actions, spatial configurations, entity interactions) with observed effects.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.episodes: List[EpisodicRecord] = []
        # Semantic correlation tracker: antecedent -> effect -> count
        self.contingency_table: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Total times antecedent was observed (for calculating empirical conditional probability)
        self.antecedent_totals: Dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self.episodes.clear()
        self.contingency_table.clear()
        self.antecedent_totals.clear()

    def record_transition(
        self,
        step: int,
        action: Action,
        prev_obs: Observation,
        next_obs: Observation,
        diff: StateTransitionDiff,
        spatial_relations: List[SpatialRelation],
        reward: float = 0.0
    ) -> None:
        rec = EpisodicRecord(
            step=step,
            action=action,
            prev_obs=prev_obs,
            next_obs=next_obs,
            diff=diff,
            reward=reward
        )
        self.episodes.append(rec)
        if len(self.episodes) > self.capacity:
            self.episodes.pop(0)

        # Update Semantic Contingency Table
        # Identify active antecedents
        active_antecedents = []

        # 1. Action direct
        active_antecedents.append(f"ACTION:{action.value}")

        # 2. Agent adjacency to entities
        for rel in spatial_relations:
            if rel.is_adjacent:
                # Map entity colors
                e1 = next((e for e in prev_obs.entities if e.id == rel.entity1_id), None)
                e2 = next((e for e in prev_obs.entities if e.id == rel.entity2_id), None)
                if e1 and e2:
                    c1, c2 = sorted([e1.color, e2.color])
                    active_antecedents.append(f"ADJACENT:{c1}:{c2}")

        # 3. Increment antecedent totals
        for ant in active_antecedents:
            self.antecedent_totals[ant] += 1

        # 4. Map antecedents to observed events (effects)
        for event in diff.events:
            for ant in active_antecedents:
                self.contingency_table[ant][event] += 1

    def get_empirical_probability(self, antecedent: str, effect: str) -> float:
        total = self.antecedent_totals.get(antecedent, 0)
        if total == 0:
            return 0.0
        return self.contingency_table[antecedent].get(effect, 0) / float(total)

    def find_records_with_event(self, event_prefix: str) -> List[EpisodicRecord]:
        results = []
        for ep in self.episodes:
            if any(ev.startswith(event_prefix) for ev in ep.diff.events):
                results.append(ep)
        return results

    def get_recent(self, k: int = 5) -> List[EpisodicRecord]:
        return self.episodes[-k:]

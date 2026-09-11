from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from world_model_lab.core.types import Observation, Entity, Position, Action

@dataclass
class SpatialRelation:
    entity1_id: str
    entity2_id: str
    distance: int
    is_adjacent: bool
    direction: str  # 'N', 'S', 'E', 'W', 'NE', etc.

@dataclass
class StateTransitionDiff:
    step: int
    action: Optional[Action]
    moved_entities: Dict[str, Tuple[Position, Position]] = field(default_factory=dict)
    state_changes: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)

class PerceptionModule:
    """
    Perception Module:
    Extracts structured symbolic entities, spatial relations (adjacency, distances),
    and temporal transition diffs between consecutive observations.
    """

    def __init__(self):
        self.last_obs: Optional[Observation] = None

    def reset(self) -> None:
        self.last_obs = None

    def extract_spatial_relations(self, obs: Observation) -> List[SpatialRelation]:
        relations: List[SpatialRelation] = []
        n = len(obs.entities)
        for i in range(n):
            for j in range(i + 1, n):
                e1 = obs.entities[i]
                e2 = obs.entities[j]
                dist = e1.pos.manhattan_distance(e2.pos)
                is_adj = e1.pos.is_adjacent(e2.pos)
                direction = self._get_relative_direction(e1.pos, e2.pos)
                relations.append(SpatialRelation(
                    entity1_id=e1.id,
                    entity2_id=e2.id,
                    distance=dist,
                    is_adjacent=is_adj,
                    direction=direction
                ))
        return relations

    def extract_diff(self, current_obs: Observation) -> StateTransitionDiff:
        diff = StateTransitionDiff(
            step=current_obs.step,
            action=current_obs.last_action,
            events=list(current_obs.events)
        )

        if self.last_obs is not None:
            # Map previous entities by id
            prev_map = {e.id: e for e in self.last_obs.entities}
            for curr_e in current_obs.entities:
                if curr_e.id in prev_map:
                    prev_e = prev_map[curr_e.id]
                    # Check position movement
                    if prev_e.pos.r != curr_e.pos.r or prev_e.pos.c != curr_e.pos.c:
                        diff.moved_entities[curr_e.id] = (
                            Position(prev_e.pos.r, prev_e.pos.c),
                            Position(curr_e.pos.r, curr_e.pos.c)
                        )
                    # Check state change
                    if prev_e.state != curr_e.state:
                        diff.state_changes[curr_e.id] = (prev_e.state, curr_e.state)

        self.last_obs = current_obs
        return diff

    def _get_relative_direction(self, p1: Position, p2: Position) -> str:
        dr = p2.r - p1.r
        dc = p2.c - p1.c
        res = ""
        if dr > 0:
            res += "S"
        elif dr < 0:
            res += "N"
        if dc > 0:
            res += "E"
        elif dc < 0:
            res += "W"
        return res or "SAME"

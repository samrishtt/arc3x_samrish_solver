from typing import List, Optional, Tuple
import numpy as np
from world_model_lab.core.types import Action, Observation, Position, Entity

class BaselineAgent:
    """
    Standard Baseline Agent (Reactive / Memory-Augmented without World Model):
    - Observes the environment.
    - Maintains a basic interaction history / visited locations.
    - Chooses actions greedily or with random exploratory steps.
    - Lacks explicit competing hypotheses, counterfactual mental simulation,
      and active disagreement-based experiment selection.
    """

    def __init__(self, seed: int = 42, exploration_rate: float = 0.3):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.exploration_rate = exploration_rate
        self.visited_positions: set = set()
        self.action_history: List[Action] = []
        self.step_count = 0

    def reset(self) -> None:
        self.visited_positions.clear()
        self.action_history.clear()
        self.step_count = 0

    def select_action(self, obs: Observation) -> Action:
        self.step_count += 1
        self.visited_positions.add((obs.agent_pos.r, obs.agent_pos.c))

        # Check if already adjacent to any interactable entity
        adj_entities = [
            e for e in obs.entities
            if not e.is_agent and obs.agent_pos.is_adjacent(e.pos)
        ]

        # Exploratory action (random)
        if self.rng.rand() < self.exploration_rate:
            return self.rng.choice(Action.all_actions())

        # If adjacent to an entity, try interacting
        if adj_entities and self.rng.rand() < 0.5:
            return Action.INTERACT

        # Otherwise, greedily move towards the closest non-agent entity
        non_agent_entities = [e for e in obs.entities if not e.is_agent and not e.is_static]
        if not non_agent_entities:
            non_agent_entities = [e for e in obs.entities if not e.is_agent]

        if non_agent_entities:
            # Pick the entity with smallest Manhattan distance
            closest = min(non_agent_entities, key=lambda e: obs.agent_pos.manhattan_distance(e.pos))
            best_action = self._step_towards(obs.agent_pos, closest.pos)
            if best_action:
                return best_action

        # Fallback: random movement
        return self.rng.choice(Action.movement_actions())

    def _step_towards(self, current: Position, target: Position) -> Optional[Action]:
        dr = target.r - current.r
        dc = target.c - current.c

        choices = []
        if dr > 0:
            choices.append(Action.DOWN)
        elif dr < 0:
            choices.append(Action.UP)

        if dc > 0:
            choices.append(Action.RIGHT)
        elif dc < 0:
            choices.append(Action.LEFT)

        if choices:
            return self.rng.choice(choices)
        return None

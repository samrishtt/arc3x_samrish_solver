from typing import List, Dict, Tuple, Optional
from enum import Enum
import numpy as np
from world_model_lab.core.types import Action, Observation, Hypothesis, Position, ConditionType
from world_model_lab.world_model.simulator import CounterfactualSimulator

class SelectionStrategy(Enum):
    RANDOM = "RANDOM"                               # Strategy A
    REACTIVE_GREEDY = "REACTIVE_GREEDY"             # Strategy B
    UNCERTAINTY_ONLY = "UNCERTAINTY_ONLY"           # Strategy C
    PREDICTIVE_DISAGREEMENT = "PREDICTIVE_DISAGREEMENT" # Strategy D
    EXPECTED_INFO_GAIN = "EXPECTED_INFO_GAIN"       # Strategy E

class ActiveExperimentSelector:
    """
    Environment-Agnostic Experiment Selector supporting Part 7 Comparison Strategies:
    - Random Exploration (A)
    - Reactive Greedy (B)
    - Uncertainty-Only (C)
    - Predictive Disagreement (D)
    - Expected Information Gain (E)
    """

    def __init__(
        self,
        simulator: CounterfactualSimulator,
        strategy: SelectionStrategy = SelectionStrategy.PREDICTIVE_DISAGREEMENT,
        confidence_threshold: float = 0.75,
        seed: int = 42
    ):
        self.simulator = simulator
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        self.rng = np.random.RandomState(seed)

    def select_action(
        self,
        obs: Observation,
        hypotheses: List[Hypothesis],
        best_hypothesis: Optional[Hypothesis],
        reward_history: Optional[List[float]] = None
    ) -> Tuple[Action, str, float]:
        """
        Returns: (selected_action, rationale_string, expected_disagreement_or_score)
        """
        # Strategy A: Pure Random Exploration
        if self.strategy == SelectionStrategy.RANDOM:
            return self.rng.choice(Action.all_actions()), "STRATEGY_RANDOM", 0.0

        # Strategy B: Reactive / Greedy
        if self.strategy == SelectionStrategy.REACTIVE_GREEDY:
            adj_entities = [e for e in obs.entities if not e.is_agent and obs.agent_pos.is_adjacent(e.pos)]
            if adj_entities:
                return Action.INTERACT, "STRATEGY_GREEDY_INTERACT", 0.0
            # Move towards closest non-agent entity
            non_agent = [e for e in obs.entities if not e.is_agent]
            if non_agent:
                closest = min(non_agent, key=lambda e: obs.agent_pos.manhattan_distance(e.pos))
                act = self._step_towards(obs, obs.agent_pos, closest.pos)
                if act:
                    return act, "STRATEGY_GREEDY_NAVIGATE", 0.0
            return self.rng.choice(Action.movement_actions()), "STRATEGY_GREEDY_FALLBACK", 0.0

        # Exploitation check: If an entity state change previously triggered positive reward,
        # and we have a verified hypothesis predicting that state change, exploit it!
        has_rewarded_transition = reward_history and any(r > 0.0 for r in reward_history)
        if has_rewarded_transition and best_hypothesis and best_hypothesis.confidence >= self.confidence_threshold:
            action = self._plan_for_hypothesis(obs, best_hypothesis)
            if action:
                return action, f"EXPLOIT: Verified rewarding rule {best_hypothesis.description()}", 0.0

        # Evaluate predictive disagreement across candidate actions
        disagreements: Dict[Action, float] = {}
        info_gains: Dict[Action, float] = {}

        # Belief entropy across hypothesis distribution
        confs = [h.confidence for h in hypotheses if h.confidence > 0]
        belief_entropy = -sum(p * np.log2(p + 1e-12) for p in confs) if confs else 0.0

        for action in Action.all_actions():
            d = self.simulator.compute_disagreement(obs, action, hypotheses)
            disagreements[action] = d
            info_gains[action] = d * belief_entropy

        # Strategy C: Uncertainty-only
        if self.strategy == SelectionStrategy.UNCERTAINTY_ONLY:
            # Picks the most uncertain hypothesis (confidence closest to 0.5) and navigates to test it
            uncertain_h = min(hypotheses, key=lambda h: abs(h.confidence - 0.5))
            act = self._plan_for_hypothesis(obs, uncertain_h)
            if act:
                return act, f"UNCERTAINTY_ONLY: Testing {uncertain_h.id}", 0.0

        # Strategy E: Expected Information Gain
        if self.strategy == SelectionStrategy.EXPECTED_INFO_GAIN:
            best_action = max(info_gains, key=info_gains.get)
            max_ig = info_gains[best_action]
            if max_ig > 0.05:
                return best_action, f"DISCRIMINATE_INFO_GAIN: Max IG = {max_ig:.3f}", max_ig

        # Strategy D: Predictive Disagreement (Default)
        best_action = max(disagreements, key=disagreements.get)
        max_disagreement = disagreements[best_action]
        if max_disagreement > 0.05:
            return best_action, f"DISCRIMINATE: Max disagreement = {max_disagreement:.3f}", max_disagreement

        # If immediate actions offer no disagreement, stage the highest-uncertainty hypothesis
        # Pick hypotheses with highest uncertainty / competitive confidence
        candidate_hypotheses = sorted(
            hypotheses,
            key=lambda h: (h.confidence, -h.falsification_count),
            reverse=True
        )

        for target_h in candidate_hypotheses:
            staged_action = self._plan_for_hypothesis(obs, target_h)
            if staged_action:
                return staged_action, f"STAGE_EXPERIMENT: Setting up test for {target_h.id}", 0.0

        # Fallback: exploratory step
        return Action.UP, "DEFAULT_EXPLORATION", 0.0

    def _plan_for_hypothesis(self, obs: Observation, h: Hypothesis) -> Optional[Action]:
        cause_ent = obs.get_entity_by_color(h.cause_color)
        if not cause_ent:
            return None

        if h.condition == ConditionType.TOUCH:
            if obs.agent_pos.is_adjacent(cause_ent.pos):
                return Action.INTERACT
            return self._step_towards(obs, obs.agent_pos, cause_ent.pos)

        elif h.condition == ConditionType.ADJACENCY:
            if not h.trigger_color:
                return None
            trig_ent = obs.get_entity_by_color(h.trigger_color)
            if not trig_ent:
                return None

            if cause_ent.is_static and trig_ent.is_static:
                return None

            movable = cause_ent if not cause_ent.is_static else trig_ent
            anchor = trig_ent if not cause_ent.is_static else cause_ent

            if movable.pos.is_adjacent(anchor.pos):
                return Action.WAIT

            return self._push_entity_towards(obs, obs.agent_pos, movable.pos, anchor.pos)

        return None

    def _find_path(self, start: Position, target: Position, obstacles: set) -> Optional[Action]:
        if start.r == target.r and start.c == target.c:
            return Action.WAIT
        queue = [(start.r, start.c, [])]
        visited = {(start.r, start.c)}
        moves = [
            (Action.UP, -1, 0),
            (Action.DOWN, 1, 0),
            (Action.LEFT, 0, -1),
            (Action.RIGHT, 0, 1)
        ]
        while queue:
            r, c, path = queue.pop(0)
            if r == target.r and c == target.c:
                return path[0] if path else Action.WAIT
            for act, dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.simulator.grid_size and 0 <= nc < self.simulator.grid_size:
                    if (nr, nc) not in visited and (nr, nc) not in obstacles:
                        visited.add((nr, nc))
                        queue.append((nr, nc, path + [act]))
        return None

    def _push_entity_towards(
        self,
        obs: Observation,
        agent_pos: Position,
        box_pos: Position,
        target_pos: Position
    ) -> Action:
        dr = target_pos.r - box_pos.r
        dc = target_pos.c - box_pos.c

        if abs(dc) >= abs(dr) and dc != 0:
            push_dir = Action.RIGHT if dc > 0 else Action.LEFT
            required_agent_pos = Position(box_pos.r, box_pos.c - 1 if dc > 0 else box_pos.c + 1)
        elif dr != 0:
            push_dir = Action.DOWN if dr > 0 else Action.UP
            required_agent_pos = Position(box_pos.r - 1 if dr > 0 else box_pos.r + 1, box_pos.c)
        else:
            return Action.WAIT

        if agent_pos.r == required_agent_pos.r and agent_pos.c == required_agent_pos.c:
            return push_dir

        static_obstacles = {(e.pos.r, e.pos.c) for e in obs.entities if e.is_static and not (e.pos.r == target_pos.r and e.pos.c == target_pos.c)}
        obstacles = static_obstacles | {(box_pos.r, box_pos.c)}
        act = self._find_path(agent_pos, required_agent_pos, obstacles)
        if act:
            return act

        return self._navigate_around(agent_pos, required_agent_pos, box_pos)

    def _navigate_around(
        self,
        current: Position,
        target: Position,
        obstacle: Position
    ) -> Action:
        dr = target.r - current.r
        dc = target.c - current.c

        candidate_moves = []
        if dr > 0:
            candidate_moves.append((Action.DOWN, Position(current.r + 1, current.c)))
        elif dr < 0:
            candidate_moves.append((Action.UP, Position(current.r - 1, current.c)))

        if dc > 0:
            candidate_moves.append((Action.RIGHT, Position(current.r, current.c + 1)))
        elif dc < 0:
            candidate_moves.append((Action.LEFT, Position(current.r, current.c - 1)))

        for act, next_pos in candidate_moves:
            if not (next_pos.r == obstacle.r and next_pos.c == obstacle.c):
                return act

        for act in [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]:
            dr_cand = -1 if act == Action.UP else (1 if act == Action.DOWN else 0)
            dc_cand = -1 if act == Action.LEFT else (1 if act == Action.RIGHT else 0)
            next_pos = Position(current.r + dr_cand, current.c + dc_cand)
            if not (next_pos.r == obstacle.r and next_pos.c == obstacle.c):
                return act

        return Action.WAIT

    def _step_towards(self, obs: Observation, current: Position, target: Position) -> Optional[Action]:
        obstacles = {(e.pos.r, e.pos.c) for e in obs.entities if not e.is_agent and not (e.pos.r == target.r and e.pos.c == target.c)}
        act = self._find_path(current, target, obstacles)
        if act:
            return act
        # If direct path to target is blocked, find path to adjacent cell
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_r, adj_c = target.r + dr, target.c + dc
            if 0 <= adj_r < self.simulator.grid_size and 0 <= adj_c < self.simulator.grid_size:
                if (adj_r, adj_c) not in obstacles:
                    act = self._find_path(current, Position(adj_r, adj_c), obstacles)
                    if act:
                        return act
        # Fallback to random movement to unstick
        return self.rng.choice(Action.movement_actions())

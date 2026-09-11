from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import copy
from world_model_lab.core.types import (
    Position, Entity, Action, Observation,
    ConditionType, EffectType, StepResult
)

# Color to integer mapping for raw grid matrix
COLOR_MAP = {
    "empty": 0,
    "agent": 1,
    "red": 2,
    "green": 3,
    "blue": 4,
    "yellow": 5,
    "purple": 6,
    "orange": 7,
    "cyan": 8,
    "wall": 9,
}

class HiddenRule:
    def __init__(
        self,
        rule_id: str,
        condition: ConditionType,
        cause_color: str,
        trigger_color: Optional[str],
        effect: EffectType,
        target_color: str,
        target_new_state: str
    ):
        self.rule_id = rule_id
        self.condition = condition
        self.cause_color = cause_color
        self.trigger_color = trigger_color
        self.effect = effect
        self.target_color = target_color
        self.target_new_state = target_new_state

    def check_and_apply(
        self,
        last_action: Action,
        agent_pos: Position,
        interacted_entity: Optional[Entity],
        entities: List[Entity]
    ) -> List[str]:
        events = []
        target = None
        for e in entities:
            if e.color == self.target_color:
                target = e
                break

        if not target:
            return events

        triggered = False

        if self.condition == ConditionType.TOUCH:
            if last_action == Action.INTERACT and interacted_entity:
                if interacted_entity.color == self.cause_color:
                    triggered = True

        elif self.condition == ConditionType.ADJACENCY:
            cause_ent = next((e for e in entities if e.color == self.cause_color), None)
            trig_ent = next((e for e in entities if e.color == self.trigger_color), None)
            if cause_ent and trig_ent:
                if cause_ent.pos.is_adjacent(trig_ent.pos):
                    triggered = True

        elif self.condition == ConditionType.CO_LOCATION:
            cause_ent = next((e for e in entities if e.color == self.cause_color), None)
            trig_ent = next((e for e in entities if e.color == self.trigger_color), None)
            if cause_ent and trig_ent:
                if cause_ent.pos.r == trig_ent.pos.r and cause_ent.pos.c == trig_ent.pos.c:
                    triggered = True

        if triggered:
            if self.effect == EffectType.STATE_CHANGE:
                if target.state != self.target_new_state:
                    target.state = self.target_new_state
                    events.append(f"STATE_CHANGE:{target.color}:{self.target_new_state}")
        else:
            # Revert state if condition is continuous adjacency
            if self.condition == ConditionType.ADJACENCY and target.state == self.target_new_state:
                target.state = "dormant"
                events.append(f"STATE_REVERT:{target.color}:dormant")

        return events

class GridLabEnv:
    def __init__(
        self,
        size: int = 7,
        max_steps: int = 100,
        rules: Optional[List[HiddenRule]] = None,
        seed: int = 42
    ):
        self.size = size
        self.max_steps = max_steps
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.step_count = 0
        self.rules = rules or []
        self.entities: List[Entity] = []
        self.agent: Optional[Entity] = None
        self.reset()

    def add_rule(self, rule: HiddenRule) -> None:
        self.rules.append(rule)

    def set_rules(self, rules: List[HiddenRule]) -> None:
        self.rules = rules

    def reset(self, seed: Optional[int] = None) -> Observation:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.RandomState(seed)

        self.step_count = 0
        self.entities = []

        # Default entities if none present
        # Place Agent at (1, 1)
        self.agent = Entity(
            id="agent",
            color="agent",
            shape="agent",
            pos=Position(1, 1),
            is_agent=True
        )
        self.entities.append(self.agent)

        # Place Red block at (3, 2)
        self.entities.append(Entity(
            id="red_block",
            color="red",
            shape="square",
            pos=Position(3, 2),
            state="normal"
        ))

        # Place Green block at (3, 4)
        self.entities.append(Entity(
            id="green_block",
            color="green",
            shape="circle",
            pos=Position(3, 4),
            state="normal"
        ))

        # Place Blue target at (5, 5)
        self.entities.append(Entity(
            id="blue_target",
            color="blue",
            shape="triangle",
            pos=Position(5, 5),
            state="dormant",
            is_static=True
        ))

        # Place Yellow target at (1, 5)
        self.entities.append(Entity(
            id="yellow_target",
            color="yellow",
            shape="diamond",
            pos=Position(1, 5),
            state="dormant",
            is_static=True
        ))

        return self._get_obs(events=["RESET"])

    def step(self, action: Action) -> StepResult:
        self.step_count += 1
        events: List[str] = []
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

            new_r = np.clip(self.agent.pos.r + dr, 0, self.size - 1)
            new_c = np.clip(self.agent.pos.c + dc, 0, self.size - 1)
            target_pos = Position(int(new_r), int(new_c))

            # Check if there is an entity at target position
            blocking_entity = self._entity_at(target_pos, exclude_agent=True)

            if blocking_entity:
                if not blocking_entity.is_static:
                    # Attempt to push
                    push_r = np.clip(blocking_entity.pos.r + dr, 0, self.size - 1)
                    push_c = np.clip(blocking_entity.pos.c + dc, 0, self.size - 1)
                    push_pos = Position(int(push_r), int(push_c))
                    push_blocker = self._entity_at(push_pos, exclude_agent=True)

                    if not push_blocker and (push_pos.r != blocking_entity.pos.r or push_pos.c != blocking_entity.pos.c):
                        blocking_entity.pos = push_pos
                        self.agent.pos = target_pos
                        events.append(f"PUSH:{blocking_entity.color}:{push_pos.r},{push_pos.c}")
                        events.append(f"MOVE:agent:{target_pos.r},{target_pos.c}")
                    else:
                        events.append(f"COLLIDE:{blocking_entity.color}")
                else:
                    events.append(f"COLLIDE_STATIC:{blocking_entity.color}")
            else:
                self.agent.pos = target_pos
                events.append(f"MOVE:agent:{target_pos.r},{target_pos.c}")

        elif action == Action.INTERACT:
            # Check entities in 4 adjacent directions or at same cell
            adj_entities = self._get_adjacent_entities(self.agent.pos)
            if adj_entities:
                interacted_entity = adj_entities[0]
                events.append(f"INTERACT:{interacted_entity.color}")
            else:
                events.append("INTERACT:empty")

        elif action == Action.WAIT:
            events.append("WAIT")

        # Evaluate hidden rules
        for rule in self.rules:
            rule_events = rule.check_and_apply(
                last_action=action,
                agent_pos=self.agent.pos,
                interacted_entity=interacted_entity,
                entities=self.entities
            )
            events.extend(rule_events)

        done = self.step_count >= self.max_steps
        obs = self._get_obs(last_action=action, events=events)

        # Reward signal: +1.0 if any target reaches 'active', small step penalty
        reward = -0.01
        for e in self.entities:
            if e.state == "active":
                reward += 1.0

        info = {
            "step": self.step_count,
            "events": events,
            "agent_pos": (self.agent.pos.r, self.agent.pos.c)
        }

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def mutate_rule(self, old_target_color: str, new_target_color: str, new_state: str = "active") -> None:
        """Rule mutation test: secretly alters the hidden rule mid-run."""
        for rule in self.rules:
            if rule.target_color == old_target_color:
                rule.target_color = new_target_color
                rule.target_new_state = new_state

    def reskin(self, color_mapping: Dict[str, str]) -> None:
        """Visual transfer test: permutes surface color tokens while preserving relational logic."""
        for e in self.entities:
            if e.color in color_mapping:
                e.color = color_mapping[e.color]
        for rule in self.rules:
            if rule.cause_color in color_mapping:
                rule.cause_color = color_mapping[rule.cause_color]
            if rule.trigger_color and rule.trigger_color in color_mapping:
                rule.trigger_color = color_mapping[rule.trigger_color]
            if rule.target_color in color_mapping:
                rule.target_color = color_mapping[rule.target_color]

    def _entity_at(self, pos: Position, exclude_agent: bool = False) -> Optional[Entity]:
        for e in self.entities:
            if exclude_agent and e.is_agent:
                continue
            if e.pos.r == pos.r and e.pos.c == pos.c:
                return e
        return None

    def _get_adjacent_entities(self, pos: Position) -> List[Entity]:
        results = []
        for e in self.entities:
            if e.is_agent:
                continue
            if pos.is_adjacent(e.pos) or (pos.r == e.pos.r and pos.c == e.pos.c):
                results.append(e)
        return results

    def _get_obs(self, last_action: Optional[Action] = None, events: Optional[List[str]] = None) -> Observation:
        raw_grid = np.zeros((self.size, self.size), dtype=np.int32)
        cloned_entities = [e.clone() for e in self.entities]

        for e in self.entities:
            val = COLOR_MAP.get(e.color, 9)
            raw_grid[e.pos.r, e.pos.c] = val

        return Observation(
            step=self.step_count,
            grid_shape=(self.size, self.size),
            entities=cloned_entities,
            agent_pos=Position(self.agent.pos.r, self.agent.pos.c),
            raw_grid=raw_grid,
            last_action=last_action,
            events=events or []
        )

    def render_ascii(self) -> str:
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        for e in self.entities:
            char = e.color[0].upper()
            if e.is_agent:
                char = "@"
            elif e.state == "active":
                char = char + "*"
            grid[e.pos.r][e.pos.c] = char.ljust(2)

        lines = ["+" + "---" * self.size + "+"]
        for r in grid:
            lines.append("| " + " ".join(r) + " |")
        lines.append("+" + "---" * self.size + "+")
        return "\n".join(lines)

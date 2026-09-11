from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

class Action(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    INTERACT = "INTERACT"  # touch or toggle object in front / at same cell
    WAIT = "WAIT"

    @classmethod
    def movement_actions(cls) -> List["Action"]:
        return [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]

    @classmethod
    def all_actions(cls) -> List["Action"]:
        return [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT, cls.INTERACT, cls.WAIT]

@dataclass
class Position:
    r: int
    c: int

    def to_tuple(self) -> Tuple[int, int]:
        return (self.r, self.c)

    def manhattan_distance(self, other: "Position") -> int:
        return abs(self.r - other.r) + abs(self.c - other.c)

    def is_adjacent(self, other: "Position") -> bool:
        return self.manhattan_distance(other) == 1

@dataclass
class Entity:
    id: str
    color: str  # e.g., 'red', 'green', 'blue', 'yellow', 'agent'
    shape: str  # 'circle', 'square', 'triangle', 'agent'
    pos: Position
    state: str = "normal"  # 'normal', 'active', 'open', 'dormant', 'glowing'
    is_agent: bool = False
    is_static: bool = False

    def clone(self) -> "Entity":
        return Entity(
            id=self.id,
            color=self.color,
            shape=self.shape,
            pos=Position(self.pos.r, self.pos.c),
            state=self.state,
            is_agent=self.is_agent,
            is_static=self.is_static
        )

@dataclass
class Observation:
    step: int
    grid_shape: Tuple[int, int]
    entities: List[Entity]
    agent_pos: Position
    raw_grid: np.ndarray
    last_action: Optional[Action] = None
    events: List[str] = field(default_factory=list)

    def get_entity_by_color(self, color: str) -> Optional[Entity]:
        for e in self.entities:
            if e.color == color and not e.is_agent:
                return e
        return None

    def get_entity_at(self, pos: Position) -> Optional[Entity]:
        for e in self.entities:
            if e.pos.r == pos.r and e.pos.c == pos.c:
                return e
        return None

class ConditionType(Enum):
    TOUCH = "TOUCH"                   # Agent directly interacts with entity
    ADJACENCY = "ADJACENCY"           # Entity A is adjacent to Entity B
    CO_LOCATION = "CO_LOCATION"       # Entity A is at same location as Entity B
    PUSH = "PUSH"                     # Agent pushes Entity A into Entity B

class EffectType(Enum):
    STATE_CHANGE = "STATE_CHANGE"     # Target flips state (e.g. dormant -> active)
    COLOR_CHANGE = "COLOR_CHANGE"     # Target changes color
    DOOR_OPEN = "DOOR_OPEN"           # Target barrier disappears/opens
    TELEPORT = "TELEPORT"             # Target or agent teleports
    NONE = "NONE"

@dataclass
class Hypothesis:
    id: str
    condition: ConditionType
    cause_color: str
    trigger_color: Optional[str]  # e.g. for ADJACENCY(cause='red', trigger='green')
    effect: EffectType
    effect_target_color: str      # e.g. 'blue'
    effect_new_state: str         # e.g. 'active'
    confidence: float = 0.5       # Bayesian-like probability / weight
    success_count: int = 0
    falsification_count: int = 0

    def description(self) -> str:
        if self.condition == ConditionType.TOUCH:
            return f"TOUCH({self.cause_color}) -> {self.effect_target_color} changes to {self.effect_new_state}"
        elif self.condition == ConditionType.ADJACENCY:
            return f"ADJACENT({self.cause_color}, {self.trigger_color}) -> {self.effect_target_color} changes to {self.effect_new_state}"
        return f"{self.condition.value}({self.cause_color}) -> {self.effect_target_color}:{self.effect_new_state}"

@dataclass
class MentalRollout:
    action: Action
    hypothesis_id: str
    predicted_events: List[str]
    predicted_target_state: str
    confidence: float

@dataclass
class PredictionError:
    step: int
    action: Action
    predicted_events: List[str]
    actual_events: List[str]
    discrepancy: bool
    faulty_hypothesis_id: Optional[str] = None
    attribution_notes: str = ""

@dataclass
class StepResult:
    obs: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

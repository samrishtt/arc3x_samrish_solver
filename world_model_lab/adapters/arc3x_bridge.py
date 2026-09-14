"""
Bridge between world_model_lab and arc3x.

Enables ARC-AGI-3 games to be solved using:
1. Multi-hypothesis world models (Avatar candidates, Goal candidates, Hazard candidates).
2. Predictive disagreement scoring over ARC-AGI-3 actions.
3. Prediction-error driven self-correction.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from world_model_lab.core.types import (
    Observation, Entity, Position, Action, Hypothesis,
    ConditionType, EffectType
)

# ARC-AGI-3 standard action mappings:
# 1: Up, 2: Down, 3: Left, 4: Right, 5: Use/Action, 6: Click/Interact, 7: Reset
ARC_ACTION_MAP = {
    1: Action.UP,
    2: Action.DOWN,
    3: Action.LEFT,
    4: Action.RIGHT,
    5: Action.INTERACT,
    6: Action.INTERACT,
    7: Action.WAIT
}

REVERSE_ARC_ACTION_MAP = {
    Action.UP: 1,
    Action.DOWN: 2,
    Action.LEFT: 3,
    Action.RIGHT: 4,
    Action.INTERACT: 5,
    Action.WAIT: 7
}

class ARC3xBridge:
    """
    Translates ARC-AGI-3 game grid frames (e.g. from GradedRun / GObs)
    into structured WorldModelLab Observations, and translates active
    world model experiment selections back into ARC-AGI-3 action integers.
    """

    def __init__(self, target_color: str = "goal"):
        self.target_color = target_color

    def frame_to_observation(
        self,
        frame: np.ndarray,
        step: int,
        avatar_pos: Optional[Tuple[int, int]] = None,
        last_action_int: Optional[int] = None,
        events: Optional[List[str]] = None
    ) -> Observation:
        """Converts an ARC-AGI-3 2D frame into a structured Observation."""
        frame = np.asarray(frame)
        if frame.ndim == 3:
            frame = frame[-1]
        elif frame.ndim == 1:
            dim = int(np.sqrt(len(frame)))
            frame = frame.reshape((dim, dim))
        h, w = frame.shape
        entities: List[Entity] = []

        # Find agent pos or use default center
        if avatar_pos is not None:
            a_pos = Position(avatar_pos[0], avatar_pos[1])
        else:
            a_pos = Position(h // 2, w // 2)

        agent_entity = Entity(
            id="arc_avatar",
            color="agent",
            shape="agent",
            pos=a_pos,
            is_agent=True
        )
        entities.append(agent_entity)

        # Segment color blobs from frame
        unique_colors = np.unique(frame)
        for c_val in unique_colors:
            if c_val == 0:  # typically background
                continue
            coords = np.argwhere(frame == c_val)
            if len(coords) > 0:
                mean_r = int(np.mean(coords[:, 0]))
                mean_c = int(np.mean(coords[:, 1]))
                ent = Entity(
                    id=f"blob_color_{c_val}",
                    color=f"color_{c_val}",
                    shape="blob",
                    pos=Position(mean_r, mean_c),
                    state="normal"
                )
                entities.append(ent)

        last_action = ARC_ACTION_MAP.get(last_action_int, Action.WAIT) if last_action_int else None

        return Observation(
            step=step,
            grid_shape=(h, w),
            entities=entities,
            agent_pos=a_pos,
            raw_grid=frame,
            last_action=last_action,
            events=events or []
        )

    def action_to_arc(self, action: Action) -> int:
        """Converts WorldModelLab Action to ARC-AGI-3 integer action."""
        return REVERSE_ARC_ACTION_MAP.get(action, 1)

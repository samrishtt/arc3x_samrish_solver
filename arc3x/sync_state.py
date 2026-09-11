"""Astra Sync State: Symbolic World Modeling, Forward Simulation, and Mental-to-World Synchronization.

Designed for ARC-AGI-3 environments to achieve zero-action mental simulation and instantaneous
rule induction.
Supports:
- Frame-to-frame difference and semantic change extraction
- SymbolicWorldModel with forward mental action prediction
- Active discrepancy blame (obstacle detection, fatal trap learning, switch toggling)
- Compact Algebraic Domain-Specific Language (DSL) generation for LLM context injection
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
import numpy as np


def to_grid(grid_or_frame) -> list[list[int]]:
    if grid_or_frame is None:
        return []
    if isinstance(grid_or_frame, np.ndarray):
        return grid_or_frame.tolist()
    if isinstance(grid_or_frame, list):
        if len(grid_or_frame) == 0:
            return []
        if isinstance(grid_or_frame[0], list):
            return grid_or_frame
        if hasattr(grid_or_frame[0], "tolist"):
            return [row.tolist() for row in grid_or_frame]
    if hasattr(grid_or_frame, "grid"):
        g = grid_or_frame.grid
        return g.tolist() if hasattr(g, "tolist") else list(g)
    if hasattr(grid_or_frame, "to_list"):
        return grid_or_frame.to_list()
    return []


def grid_diff(frame_a, frame_b) -> dict[tuple[int, int], tuple[int, int]]:
    """Returns mapping (r, c) -> (old_val, new_val) for all cells differing between frames."""
    ga = to_grid(frame_a)
    gb = to_grid(frame_b)
    if not ga or not gb:
        return {}

    rows = min(len(ga), len(gb))
    diffs = {}
    for r in range(rows):
        cols = min(len(ga[r]), len(gb[r]))
        for c in range(cols):
            if ga[r][c] != gb[r][c]:
                diffs[(r, c)] = (ga[r][c], gb[r][c])
    return diffs


@dataclass
class SymbolicWorldModel:
    """Compact symbolic world model representing ARC-AGI game mechanics as logical rules."""

    background: int = 0
    avatar_color: int = -1
    body_colors: set[int] = field(default_factory=set)
    avatar_pos: tuple[int, int] | None = None
    tile_size: int = 1

    # Action deltas: "ACTION1" -> (-1, 0)
    deltas: dict[str, tuple[int, int]] = field(default_factory=lambda: {
        "ACTION1": (-1, 0),
        "ACTION2": (1, 0),
        "ACTION3": (0, -1),
        "ACTION4": (0, 1),
    })

    # Evidence counters
    walkable_evidence: Counter = field(default_factory=Counter)
    blocked_evidence: Counter = field(default_factory=Counter)
    fatal_evidence: Counter = field(default_factory=Counter)
    goal_evidence: Counter = field(default_factory=Counter)

    # Current grid state
    grid: list[list[int]] = field(default_factory=list)

    # Mental prediction
    predicted_pos: tuple[int, int] | None = None
    last_intended_action: str | None = None

    # Step count
    level_id: int = 0
    step_count: int = 0

    @property
    def walkable_set(self) -> set[int]:
        """Colors proven walkable where walkability outweighs refusals."""
        return {
            c for c, n in self.walkable_evidence.items()
            if n > self.blocked_evidence.get(c, 0) and c not in self.fatal_set
        } or {self.background}

    @property
    def fatal_set(self) -> set[int]:
        return {c for c, n in self.fatal_evidence.items() if n >= 1}

    @property
    def blocked_set(self) -> set[int]:
        return {
            c for c, n in self.blocked_evidence.items()
            if n >= self.walkable_evidence.get(c, 0)
        }

    def init_from_frame(self, frame, avatar_hint: tuple[int, int] | None = None) -> None:
        """Initializes model state from first observation."""
        self.grid = to_grid(frame)
        if not self.grid:
            return

        rows = len(self.grid)
        cols = len(self.grid[0]) if rows > 0 else 0

        # Background = most frequent color along borders
        border_pixels = []
        for r in range(rows):
            border_pixels.append(self.grid[r][0])
            border_pixels.append(self.grid[r][cols - 1])
        for c in range(cols):
            border_pixels.append(self.grid[0][c])
            border_pixels.append(self.grid[rows - 1][c])

        if border_pixels:
            self.background = Counter(border_pixels).most_common(1)[0][0]
        self.walkable_evidence[self.background] += 5

        # If hint provided
        if avatar_hint and 0 <= avatar_hint[0] < rows and 0 <= avatar_hint[1] < cols:
            self.avatar_pos = avatar_hint
            self.avatar_color = self.grid[avatar_hint[0]][avatar_hint[1]]
            self.body_colors = {self.avatar_color}

    def predict_action(self, action: str) -> tuple[int, int] | None:
        """Forward mental simulation of action without environment execution.
        
        Predicts next avatar position based on current beliefs.
        """
        self.last_intended_action = action
        if self.avatar_pos is None:
            return None

        delta = self.deltas.get(action)
        if not delta:
            self.predicted_pos = self.avatar_pos
            return self.avatar_pos

        nr = self.avatar_pos[0] + delta[0] * self.tile_size
        nc = self.avatar_pos[1] + delta[1] * self.tile_size

        rows = len(self.grid)
        cols = len(self.grid[0]) if rows > 0 else 0

        # Boundary check
        if not (0 <= nr < rows and 0 <= nc < cols):
            self.predicted_pos = self.avatar_pos
            return self.avatar_pos

        target_color = self.grid[nr][nc]
        # Wall / obstacle check
        if target_color in self.blocked_set or target_color in self.fatal_set:
            self.predicted_pos = self.avatar_pos
            return self.avatar_pos

        self.predicted_pos = (nr, nc)
        return self.predicted_pos

    def sync(
        self,
        actual_frame,
        intended_action: str | None = None,
        level_up: bool = False,
        game_over: bool = False,
    ) -> dict[str, Any]:
        """Synchronizes mental simulation with actual observed frame, updating rules immediately."""
        new_grid = to_grid(actual_frame)
        diffs = grid_diff(self.grid, new_grid)
        self.grid = new_grid
        self.step_count += 1
        action = intended_action or self.last_intended_action

        diag = {
            "discrepancy": False,
            "avatar_moved": False,
            "blamed_rule": None,
            "actions_taken": self.step_count,
        }

        if not diffs:
            # Nothing changed on screen
            if action and self.avatar_pos and action in self.deltas:
                # Refused move: blame obstacle in the direction of delta
                dr, dc = self.deltas[action]
                nr, nc = self.avatar_pos[0] + dr, self.avatar_pos[1] + dc
                if 0 <= nr < len(self.grid) and 0 <= nc < len(self.grid[0]):
                    refused_col = self.grid[nr][nc]
                    self.blocked_evidence[refused_col] += 1
                    diag["discrepancy"] = True
                    diag["blamed_rule"] = f"refused_obstacle_color_{refused_col}"
            return diag

        # Locate avatar after action
        new_pos = None
        if self.avatar_color >= 0:
            for (r, c), (old_v, new_v) in diffs.items():
                if new_v == self.avatar_color:
                    new_pos = (r, c)
                    break

        if new_pos is not None:
            if self.avatar_pos is not None:
                # Avatar moved
                diag["avatar_moved"] = True
                # Record what it stood on (from old grid)
                stood_on = diffs.get(new_pos, (self.background, self.avatar_color))[0]
                if stood_on != self.avatar_color:
                    self.walkable_evidence[stood_on] += 2

                # Calibrate delta if single action
                if action:
                    dr = new_pos[0] - self.avatar_pos[0]
                    dc = new_pos[1] - self.avatar_pos[1]
                    if abs(dr) + abs(dc) > 0:
                        self.deltas[action] = (dr, dc)

            self.avatar_pos = new_pos

        # Check mental discrepancy
        if self.predicted_pos is not None and self.avatar_pos is not None:
            if self.avatar_pos != self.predicted_pos:
                diag["discrepancy"] = True
                diag["mental_predicted"] = self.predicted_pos
                diag["world_actual"] = self.avatar_pos

        # Game over handling
        if game_over:
            if self.avatar_pos and action in self.deltas:
                dr, dc = self.deltas[action]
                nr, nc = self.avatar_pos[0] + dr, self.avatar_pos[1] + dc
                if 0 <= nr < len(self.grid) and 0 <= nc < len(self.grid[0]):
                    fatal_col = self.grid[nr][nc]
                    self.fatal_evidence[fatal_col] += 3
                    diag["blamed_rule"] = f"fatal_trap_color_{fatal_col}"

        # Level up handling
        if level_up:
            self.level_id += 1
            for (r, c), (old_v, new_v) in diffs.items():
                if old_v != self.background and old_v != self.avatar_color:
                    self.goal_evidence[old_v] += 5
            diag["blamed_rule"] = "level_complete_goal_reached"

        return diag

    def to_dsl(self) -> str:
        """Generates compact Astra-style Algebraic DSL representation of world model state.
        
        Example:
          L0: av=3@(12,4) bg=0 tile=1 mv[1:(-1,0),2:(1,0)] walk=[0,5] block=[1] fatal=[8] goal=[2]
        """
        pos_str = f"@({self.avatar_pos[0]},{self.avatar_pos[1]})" if self.avatar_pos else "@unknown"
        mv_str = ",".join(f"{k[-1]}:{v[0]},{v[1]}" for k, v in sorted(self.deltas.items()) if k.startswith("ACTION"))
        walk_str = str(sorted(self.walkable_set))
        block_str = str(sorted(self.blocked_set))
        fatal_str = str(sorted(self.fatal_set))
        goal_str = str([c for c, _ in self.goal_evidence.most_common(2)])

        return (
            f"[DSL:WORLD_MODEL] L{self.level_id}: av={self.avatar_color}{pos_str} bg={self.background} "
            f"tile={self.tile_size} mv[{mv_str}] walk={walk_str} block={block_str} fatal={fatal_str} goal={goal_str}"
        )

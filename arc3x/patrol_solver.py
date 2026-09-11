"""Astra Patrol Solver: Moving Guard Tracking, Periodic Orbit Induction, and Space-Time (r, c, t) Planning.

Designed for ARC-AGI-3 environments featuring dynamic obstacles and moving guards (e.g. tu93, lp85).
Supports:
- Frame-to-frame non-avatar moving entity tracking
- Periodic orbit cycle detection (period T)
- Forward position projection at any future timestep t
- 3D Space-Time BFS / A* navigation with collision & swap avoidance
"""

from collections import deque
from typing import Callable, Iterable, Sequence, Union
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


def detect_period(trajectory: Sequence[tuple[int, int]]) -> tuple[int, list[tuple[int, int]]]:
    """Detects periodic cycle length T in a sequence of coordinates.
    
    Returns:
        (period_T, cycle_waypoints)
    """
    n = len(trajectory)
    if n <= 1:
        return (1, list(trajectory))

    # Test candidate periods from 1 up to n // 2
    for p in range(1, (n // 2) + 1):
        is_periodic = True
        for i in range(n - p):
            if trajectory[i] != trajectory[i + p]:
                is_periodic = False
                break
        if is_periodic:
            return (p, list(trajectory[:p]))

    # Check oscillation between two ends (e.g. 0, 1, 2, 1, 0, 1, 2...)
    for p in range(2, n):
        # Repeat prefix and compare with suffix
        prefix = trajectory[:p]
        matches = 0
        for i in range(n):
            if trajectory[i] == prefix[i % p]:
                matches += 1
        if matches >= int(0.85 * n) and matches >= 4:
            return (p, list(prefix))

    return (n, list(trajectory))


def track_moving_entities(
    history_frames: Sequence,
    avatar_boxes: Sequence[tuple[int, int, int, int] | None] | None = None,
    background: int = 0,
) -> list[dict]:
    """Extracts non-avatar moving entities across a sequence of consecutive frames."""
    if len(history_frames) < 2:
        return []

    grids = [to_grid(f) for f in history_frames]
    rows = len(grids[0])
    cols = len(grids[0][0]) if rows > 0 else 0

    # Locate dynamic cells that change between frames
    entity_history: dict[int, list[tuple[int, int]]] = {}

    for t, grid in enumerate(grids):
        av_box = avatar_boxes[t] if avatar_boxes and t < len(avatar_boxes) else None
        av_cells = set()
        if av_box:
            at, al, ah, aw = av_box
            for r in range(at, at + ah):
                for c in range(al, al + aw):
                    av_cells.add((r, c))

        # Find connected components of non-background, non-avatar cells
        visited = [[False] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                if visited[r][c] or (r, c) in av_cells:
                    continue
                val = grid[r][c]
                visited[r][c] = True
                if val == background:
                    continue

                cells = [(r, c)]
                q = deque([(r, c)])
                while q:
                    cr, cc = q.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                            if (nr, nc) not in av_cells and grid[nr][nc] == val:
                                visited[nr][nc] = True
                                cells.append((nr, nc))
                                q.append((nr, nc))

                # Track centroid of this object
                center = (sum(x[0] for x in cells) // len(cells), sum(x[1] for x in cells) // len(cells))
                entity_history.setdefault(val, []).append(center)

    results = []
    for color, centers in entity_history.items():
        if len(centers) >= 2:
            # Check if centers actually move
            if len(set(centers)) > 1:
                period, cycle = detect_period(centers)
                results.append({
                    "color": color,
                    "trajectory": centers,
                    "period": period,
                    "cycle": cycle,
                    "last_pos": centers[-1],
                })

    return results


def predict_guard_positions_at_t(
    guards: Sequence[dict],
    future_t: int,
) -> set[tuple[int, int]]:
    """Projects guard coordinates at time step future_t using their periodic cycles."""
    positions = set()
    for g in guards:
        cycle = g.get("cycle")
        if not cycle:
            continue
        p = len(cycle)
        pos = cycle[future_t % p]
        positions.add(pos)
    return positions


def space_time_plan(
    grid_or_frame,
    start: tuple[int, int],
    goal: Union[tuple[int, int], Iterable[tuple[int, int]]],
    walkable: Iterable[int] | None = None,
    guards: Sequence[dict] | None = None,
    max_timesteps: int = 120,
    allow_wait: bool = True,
    step: int = 1,
    deltas_map: dict[str, tuple[int, int]] | None = None,
) -> list[str]:
    """3D Space-Time BFS pathfinder: avoids both static walls and dynamic moving guards.
    
    Guarantees:
    - No position collision at timestep t
    - No edge swap collision (guard and avatar swapping cells in 1 step)
    - Optimal timed arrival sequence
    """
    grid = to_grid(grid_or_frame)
    if not grid or len(grid) == 0:
        return []

    rows, cols = len(grid), max(len(r) for r in grid)
    sr, sc = int(start[0]), int(start[1])

    if isinstance(goal, (tuple, list)) and len(goal) == 2 and isinstance(goal[0], (int, float)):
        goal_set = {(int(goal[0]), int(goal[1]))}
    else:
        goal_set = {(int(g[0]), int(g[1])) for g in goal}

    if (sr, sc) in goal_set:
        return []

    w_set = set(walkable) if walkable is not None else None

    s = max(1, int(step))
    if deltas_map:
        moves = dict(deltas_map)
    else:
        moves = {
            "ACTION1": (-s, 0),
            "ACTION2": (s, 0),
            "ACTION3": (0, -s),
            "ACTION4": (0, s),
        }

    # Precalculate guard positions for t = 0 .. max_timesteps
    guards_list = list(guards) if guards else []
    guard_pos_by_t = [predict_guard_positions_at_t(guards_list, t) for t in range(max_timesteps + 2)]

    # State: (r, c, t)
    # Queue: (r, c, t, actions_so_far)
    visited = {(sr, sc, 0)}
    queue = deque([(sr, sc, 0, [])])
    target_actions = None

    while queue:
        cr, cc, t, actions = queue.popleft()
        if (cr, cc) in goal_set:
            target_actions = actions
            break

        if t >= max_timesteps:
            continue

        next_t = t + 1
        guards_next = guard_pos_by_t[next_t]
        guards_curr = guard_pos_by_t[t]

        # 1. Directional moves
        for act_name, (dr, dc) in moves.items():
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                val = grid[nr][nc]
                if w_set is not None and val not in w_set and (nr, nc) not in goal_set:
                    continue

                # Collision checks:
                # Direct collision at next_t
                if (nr, nc) in guards_next:
                    continue
                # Swap collision: avatar moving into guard's old cell while guard moved into avatar's old cell
                if (nr, nc) in guards_curr and (cr, cc) in guards_next:
                    continue

                state_key = (nr, nc, next_t)
                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((nr, nc, next_t, actions + [act_name]))

        # 2. Wait in place (if permitted and safe)
        if allow_wait:
            if (cr, cc) not in guards_next:
                state_key = (cr, cc, next_t)
                if state_key not in visited:
                    visited.add(state_key)
                    # Use WAIT or ACTION5 if supported
                    queue.append((cr, cc, next_t, actions + ["WAIT"]))

    return target_actions if target_actions is not None else []

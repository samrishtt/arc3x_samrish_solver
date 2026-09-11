"""Astra Sokoban & Inventory Solver: Block Pushing, Crate Carrying, and Orientation Tracking.

Designed for ARC-AGI-3 environments featuring pushable blocks, carry/lift states, and orientation (e.g. wa30).
Supports:
- Pushable crate detection and goal receptacle matching
- Player orientation tracking (facing north, south, east, west)
- Carry/lift inventory state representation (empty vs carrying)
- Push-path BFS search (calculates where player must stand to push a box to target)
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


def find_pushable_crates(
    grid_or_frame,
    walkable_colors: Iterable[int] | None = None,
    avatar_pos: tuple[int, int] | None = None,
) -> list[dict]:
    """Identifies compact blocks that can be pushed or carried."""
    grid = to_grid(grid_or_frame)
    if not grid or len(grid) == 0:
        return []

    rows, cols = len(grid), max(len(r) for r in grid)
    w_set = set(walkable_colors) if walkable_colors else {0}

    crates = []
    visited = [[False] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            val = grid[r][c]
            if val in w_set or visited[r][c]:
                continue
            if avatar_pos and (r, c) == avatar_pos:
                continue

            # Segment connected component
            visited[r][c] = True
            cells = [(r, c)]
            q = [(r, c)]
            idx = 0
            while idx < len(q):
                cr, cc = q[idx]
                idx += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                        if grid[nr][nc] == val:
                            visited[nr][nc] = True
                            cells.append((nr, nc))
                            q.append((nr, nc))

            # Small to medium compact objects are likely crates/pushables
            if 1 <= len(cells) <= 16:
                cr = sum(x[0] for x in cells) // len(cells)
                cc = sum(x[1] for x in cells) // len(cells)
                crates.append({
                    "color": val,
                    "size": len(cells),
                    "center": (cr, cc),
                    "cells": cells,
                })

    return crates


def push_plan(
    grid_or_frame,
    avatar_start: tuple[int, int],
    crate_start: tuple[int, int],
    crate_goal: tuple[int, int],
    walkable_colors: Iterable[int] | None = None,
) -> list[str]:
    """Computes action plan to maneuver avatar behind crate and push it to crate_goal.
    
    State space: (avatar_pos, crate_pos)
    """
    grid = to_grid(grid_or_frame)
    if not grid or len(grid) == 0:
        return []

    rows, cols = len(grid), max(len(r) for r in grid)
    w_set = set(walkable_colors) if walkable_colors else {0}

    moves = {
        (-1, 0): "ACTION1",
        (1, 0): "ACTION2",
        (0, -1): "ACTION3",
        (0, 1): "ACTION4",
    }

    ar, ac = int(avatar_start[0]), int(avatar_start[1])
    kr, kc = int(crate_start[0]), int(crate_start[1])
    gr, gc = int(crate_goal[0]), int(crate_goal[1])

    # State: (ar, ac, kr, kc)
    start_state = (ar, ac, kr, kc)
    visited = {start_state}
    queue = deque([(start_state, [])])

    while queue:
        (car, cac, ckr, ckc), path = queue.popleft()
        if (ckr, ckc) == (gr, gc):
            return path

        for (dr, dc), act_name in moves.items():
            nar, nac = car + dr, cac + dc
            if not (0 <= nar < rows and 0 <= nac < cols):
                continue

            # Did player push into the crate?
            if (nar, nac) == (ckr, ckc):
                # Crate is pushed to (nkr, nkc)
                nkr, nkc = ckr + dr, ckc + dc
                if not (0 <= nkr < rows and 0 <= nkc < cols):
                    continue
                if grid[nkr][nkc] not in w_set and (nkr, nkc) != (gr, gc):
                    continue
                next_state = (nar, nac, nkr, nkc)
            else:
                # Player moves freely; crate stays
                if grid[nar][nac] not in w_set:
                    continue
                next_state = (nar, nac, ckr, ckc)

            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [act_name]))

    return []

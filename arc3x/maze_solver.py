"""Astra Maze Solver: Symbolic Pathfinding, Lattice Navigation, and Topological Analysis.

Designed for ARC-AGI-3 environments to achieve optimal action efficiency.
Supports:
- BFS/A* shortest path with arbitrary step sizes (lattice grids)
- Action sequence generation (walk_to) matching ARC-AGI convention
- Connected component object extraction and centroid detection
- Topological maze analysis (corridors, dead ends, junctions)
- Frontier exploration for unknown rooms
"""

from collections import deque
from typing import Callable, Iterable, Sequence, Union
import numpy as np


def to_grid(grid_or_frame) -> list[list[int]]:
    """Converts input frame, numpy array, or nested list to 2D list of ints."""
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


def find_path(
    grid_or_frame,
    start: tuple[int, int],
    goal: Union[tuple[int, int], Iterable[tuple[int, int]], Callable[[int, int, int], bool]],
    walkable: Union[Iterable[int], Callable[[int, int, int], bool], None] = None,
    step: int = 1,
    deltas: dict[str, tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """BFS shortest path from start to goal.
    
    Args:
        grid_or_frame: 2D grid representation
        start: (row, col) start coordinates
        goal: (row, col) target, collection of targets, or predicate(r, c, val) -> bool
        walkable: collection of walkable color ints, or predicate(r, c, val) -> bool
        step: displacement per move (default 1)
        deltas: optional custom movement deltas {name: (dr, dc)}
    
    Returns:
        List of coordinates [(r0, c0), (r1, c1), ...] from start to goal inclusive.
        Returns empty list [] if no path exists.
    """
    grid = to_grid(grid_or_frame)
    if not grid or len(grid) == 0 or len(grid[0]) == 0:
        return []

    rows, cols = len(grid), len(grid[0])
    sr, sc = int(start[0]), int(start[1])
    if not (0 <= sr < rows and 0 <= sc < cols):
        return []

    if callable(goal):
        is_goal = goal
    elif isinstance(goal, (tuple, list)) and len(goal) == 2 and isinstance(goal[0], (int, float)):
        gr, gc = int(goal[0]), int(goal[1])
        is_goal = lambda r, c, _: (r == gr and c == gc)
    else:
        goal_set = {(int(g[0]), int(g[1])) for g in goal}
        is_goal = lambda r, c, _: (r, c) in goal_set

    val_start = grid[sr][sc] if sc < len(grid[sr]) else 0
    if is_goal(sr, sc, val_start):
        return [(sr, sc)]

    if callable(walkable):
        is_walkable = walkable
    elif walkable is not None:
        w_set = set(walkable)
        is_walkable = lambda r, c, val: val in w_set
    else:
        is_walkable = lambda r, c, val: True

    if deltas:
        move_vectors = list(deltas.values())
    else:
        s = max(1, int(step))
        move_vectors = [(-s, 0), (s, 0), (0, -s), (0, s)]

    visited = { (sr, sc) }
    parent = {}
    queue = deque([(sr, sc)])
    target_reached = None

    while queue:
        cr, cc = queue.popleft()
        c_val = grid[cr][cc] if cc < len(grid[cr]) else 0
        if is_goal(cr, cc, c_val) and (cr, cc) != (sr, sc):
            target_reached = (cr, cc)
            break

        for dr, dc in move_vectors:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                n_val = grid[nr][nc] if nc < len(grid[nr]) else 0
                if is_goal(nr, nc, n_val) or is_walkable(nr, nc, n_val):
                    visited.add((nr, nc))
                    parent[(nr, nc)] = (cr, cc)
                    queue.append((nr, nc))
                    if is_goal(nr, nc, n_val):
                        target_reached = (nr, nc)
                        break
        if target_reached:
            break

    if not target_reached:
        return []

    path = [target_reached]
    curr = target_reached
    while curr != (sr, sc):
        curr = parent.get(curr)
        if curr is None:
            return []
        path.append(curr)
    path.reverse()
    return path


def walk_to(
    grid_or_frame,
    start: tuple[int, int],
    goal: Union[tuple[int, int], Iterable[tuple[int, int]], Callable[[int, int, int], bool]],
    walkable: Union[Iterable[int], Callable[[int, int, int], bool], None] = None,
    step: int = 1,
    deltas_map: dict[str, tuple[int, int]] | None = None,
) -> list[str]:
    """Translates shortest path into concrete ARC-AGI-3 action strings.
    
    Default mapping:
      (-step, 0) -> 'ACTION1' (North)
      (step, 0)  -> 'ACTION2' (South)
      (0, -step) -> 'ACTION3' (West)
      (0, step)  -> 'ACTION4' (East)
    """
    s = max(1, int(step))
    if deltas_map:
        vec_to_action = {vec: act for act, vec in deltas_map.items()}
    else:
        vec_to_action = {
            (-s, 0): "ACTION1",
            (s, 0): "ACTION2",
            (0, -s): "ACTION3",
            (0, s): "ACTION4",
        }

    path = find_path(grid_or_frame, start, goal, walkable=walkable, step=s, deltas=deltas_map)
    if len(path) < 2:
        return []

    actions = []
    for i in range(len(path) - 1):
        dr = path[i + 1][0] - path[i][0]
        dc = path[i + 1][1] - path[i][1]
        act = vec_to_action.get((dr, dc))
        if act:
            actions.append(act)
    return actions


def find_objects(grid_or_frame, ignore_colors=None) -> list[dict]:
    """Segments connected components with colors, sizes, bounding boxes and centers."""
    grid = to_grid(grid_or_frame)
    if not grid or len(grid) == 0:
        return []

    rows, cols = len(grid), max(len(r) for r in grid)
    ignore = {ignore_colors} if isinstance(ignore_colors, int) else (set(ignore_colors) if ignore_colors else set())
    visited = [[False] * cols for _ in range(rows)]
    objects = []

    for r in range(rows):
        for c in range(cols):
            if visited[r][c]:
                continue
            color = grid[r][c] if c < len(grid[r]) else None
            visited[r][c] = True
            if color is None or color in ignore:
                continue

            cells = [(r, c)]
            q = deque([(r, c)])
            min_r, max_r, min_c, max_c = r, r, c, c

            while q:
                cr, cc = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                        n_col = grid[nr][nc] if nc < len(grid[nr]) else None
                        if n_col == color:
                            visited[nr][nc] = True
                            cells.append((nr, nc))
                            q.append((nr, nc))
                            if nr < min_r: min_r = nr
                            if nr > max_r: max_r = nr
                            if nc < min_c: min_c = nc
                            if nc > max_c: max_c = nc

            center = (sum(x[0] for x in cells) // len(cells), sum(x[1] for x in cells) // len(cells))
            objects.append({
                "color": color,
                "size": len(cells),
                "center": center,
                "bbox": (min_r, min_c, max_r, max_c),
                "cells": cells,
            })

    objects.sort(key=lambda o: o["size"], reverse=True)
    return objects


def find_click_targets(grid_or_frame, bg_color=None) -> list[tuple[int, int]]:
    """Returns ranked candidate coordinates for ACTION6 click interactions."""
    objs = find_objects(grid_or_frame, ignore_colors=({bg_color} if bg_color is not None else {0}))
    if not objs:
        return []

    targets = []
    for o in objs:
        if o["size"] <= 4:
            targets.append(o["center"])
    for o in objs:
        if 4 < o["size"] <= 25:
            targets.append(o["center"])
    for o in objs:
        if o["size"] > 25:
            targets.extend([o["center"], (o["bbox"][0], o["bbox"][1]), (o["bbox"][2], o["bbox"][3])])

    seen = set()
    dedup = []
    for coord in targets:
        if coord not in seen:
            seen.add(coord)
            dedup.append(coord)
    return dedup


def find_corridors_and_junctions(grid_or_frame, walkable: Iterable[int] | None = None) -> dict[str, list[tuple[int, int]]]:
    """Analyzes maze topology: dead_ends, corridors, junctions, open_rooms."""
    grid = to_grid(grid_or_frame)
    if not grid or len(grid) == 0:
        return {"dead_ends": [], "corridors": [], "junctions": [], "open_rooms": []}

    rows, cols = len(grid), max(len(r) for r in grid)
    w_set = set(walkable) if walkable is not None else None

    is_w = lambda r, c: (0 <= r < rows and 0 <= c < cols and (w_set is None or (grid[r][c] in w_set)))

    dead_ends = []
    corridors = []
    junctions = []
    open_rooms = []

    for r in range(rows):
        for c in range(cols):
            if not is_w(r, c):
                continue
            neighbors = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if is_w(r + dr, c + dc):
                    neighbors += 1

            if neighbors == 1:
                dead_ends.append((r, c))
            elif neighbors == 2:
                corridors.append((r, c))
            elif neighbors >= 3:
                junctions.append((r, c))
            elif neighbors == 4:
                open_rooms.append((r, c))

    return {
        "dead_ends": dead_ends,
        "corridors": corridors,
        "junctions": junctions,
        "open_rooms": open_rooms,
    }


def explore_frontier(grid_or_frame, start: tuple[int, int], walkable: Iterable[int] | None = None) -> tuple[int, int] | None:
    """Finds the closest unexplored/boundary walkable cell from start."""
    grid = to_grid(grid_or_frame)
    if not grid or len(grid) == 0:
        return None

    topol = find_corridors_and_junctions(grid, walkable)
    cands = topol["junctions"] + topol["dead_ends"]
    if not cands:
        return None

    best_dist = float("inf")
    best_cand = None
    sr, sc = start
    for cr, cc in cands:
        d = abs(cr - sr) + abs(cc - sc)
        if 0 < d < best_dist:
            best_dist = d
            best_cand = (cr, cc)
    return best_cand

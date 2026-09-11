"""Astra Combat & Hazard Solver: Threat Modeling, Hazard Fields, and Safe Navigation.

Designed for ARC-AGI-3 environments featuring lethal hazards, fatal colors, and moving enemies.
Supports:
- Threat & hazard entity detection
- Smooth threat cost field computation (danger gradients)
- Safe-path A* planning with danger avoidance
- Evasion vector calculation for escaping approaching enemies
- Forward action safety verification
"""

import heapq
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


def detect_threats(
    grid_or_frame,
    fatal_colors: Iterable[int] | None = None,
    enemy_colors: Iterable[int] | None = None,
) -> list[dict]:
    """Detects all fatal hazards and enemy components on the board."""
    grid = to_grid(grid_or_frame)
    if not grid or len(grid) == 0:
        return []

    rows, cols = len(grid), max(len(r) for r in grid)
    f_set = set(fatal_colors) if fatal_colors else set()
    e_set = set(enemy_colors) if enemy_colors else set()

    threats = []
    visited = [[False] * cols for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            val = grid[r][c] if c < len(grid[r]) else None
            if val is None or visited[r][c]:
                continue

            t_type = None
            if val in f_set:
                t_type = "fatal"
            elif val in e_set:
                t_type = "enemy"

            if t_type is not None:
                # Component extraction
                visited[r][c] = True
                cells = [(r, c)]
                q = [(r, c)]
                min_r, max_r, min_c, max_c = r, r, c, c

                idx = 0
                while idx < len(q):
                    cr, cc = q[idx]
                    idx += 1
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                            n_val = grid[nr][nc] if nc < len(grid[nr]) else None
                            if n_val == val:
                                visited[nr][nc] = True
                                cells.append((nr, nc))
                                q.append((nr, nc))
                                if nr < min_r: min_r = nr
                                if nr > max_r: max_r = nr
                                if nc < min_c: min_c = nc
                                if nc > max_c: max_c = nc

                threats.append({
                    "type": t_type,
                    "color": val,
                    "cells": cells,
                    "center": (sum(x[0] for x in cells) // len(cells), sum(x[1] for x in cells) // len(cells)),
                    "bbox": (min_r, min_c, max_r, max_c),
                })
    return threats


def compute_threat_field(
    grid_or_frame,
    fatal_colors: Iterable[int] | None = None,
    enemy_colors: Iterable[int] | None = None,
    danger_radius: int = 2,
    base_threat_weight: float = 20.0,
) -> list[list[float]]:
    """Generates a 2D threat cost matrix.
    
    Lethal cells receive float('inf').
    Nearby cells within danger_radius receive decaying proximity penalties.
    """
    grid = to_grid(grid_or_frame)
    if not grid or len(grid) == 0:
        return []

    rows, cols = len(grid), max(len(r) for r in grid)
    threat_field = [[0.0] * cols for _ in range(rows)]

    f_set = set(fatal_colors) if fatal_colors else set()
    e_set = set(enemy_colors) if enemy_colors else set()

    threat_cells = []
    for r in range(rows):
        for c in range(cols):
            val = grid[r][c] if c < len(grid[r]) else 0
            if val in f_set:
                threat_field[r][c] = float("inf")
                threat_cells.append((r, c, "fatal"))
            elif val in e_set:
                threat_field[r][c] = base_threat_weight * 2.0
                threat_cells.append((r, c, "enemy"))

    # Spread threat gradient
    rad = max(1, int(danger_radius))
    for tr, tc, t_type in threat_cells:
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                nr, nc = tr + dr, tc + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if threat_field[nr][nc] == float("inf"):
                        continue
                    dist = abs(dr) + abs(dc)
                    if dist <= rad and dist > 0:
                        penalty = (base_threat_weight / dist) * (1.5 if t_type == "enemy" else 1.0)
                        threat_field[nr][nc] += penalty

    return threat_field


def safe_path(
    grid_or_frame,
    start: tuple[int, int],
    goal: Union[tuple[int, int], Iterable[tuple[int, int]]],
    walkable: Iterable[int] | None = None,
    fatal_colors: Iterable[int] | None = None,
    threat_field: list[list[float]] | None = None,
    step: int = 1,
    deltas: dict[str, tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """A* pathfinder that minimizes total path distance + threat penalty."""
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
        return [(sr, sc)]

    w_set = set(walkable) if walkable is not None else None
    f_set = set(fatal_colors) if fatal_colors else set()

    if threat_field is None:
        threat_field = compute_threat_field(grid, fatal_colors=f_set, danger_radius=2)

    s = max(1, int(step))
    move_vectors = list(deltas.values()) if deltas else [(-s, 0), (s, 0), (0, -s), (0, s)]

    def heuristic(r, c) -> float:
        return min(abs(r - gr) + abs(c - gc) for gr, gc in goal_set)

    # Priority queue: (f_score, g_score, (r, c))
    pq = [(heuristic(sr, sc), 0.0, (sr, sc))]
    cost_so_far = {(sr, sc): 0.0}
    parent = {}
    target_hit = None

    while pq:
        _, g, curr = heapq.heappop(pq)
        cr, cc = curr

        if curr in goal_set:
            target_hit = curr
            break

        if g > cost_so_far.get(curr, float("inf")):
            continue

        for dr, dc in move_vectors:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                n_val = grid[nr][nc] if nc < len(grid[nr]) else 0
                if (nr, nc) not in goal_set:
                    if n_val in f_set:
                        continue
                    if w_set is not None and n_val not in w_set:
                        continue

                threat_cost = threat_field[nr][nc] if threat_field else 0.0
                if threat_cost == float("inf"):
                    continue

                new_cost = g + 1.0 + threat_cost
                if new_cost < cost_so_far.get((nr, nc), float("inf")):
                    cost_so_far[(nr, nc)] = new_cost
                    f_score = new_cost + heuristic(nr, nc)
                    parent[(nr, nc)] = curr
                    heapq.heappush(pq, (f_score, new_cost, (nr, nc)))

    if not target_hit:
        return []

    path = [target_hit]
    curr = target_hit
    while curr != (sr, sc):
        curr = parent[curr]
        path.append(curr)
    path.reverse()
    return path


def safe_walk_to(
    grid_or_frame,
    start: tuple[int, int],
    goal: Union[tuple[int, int], Iterable[tuple[int, int]]],
    walkable: Iterable[int] | None = None,
    fatal_colors: Iterable[int] | None = None,
    threat_field: list[list[float]] | None = None,
    step: int = 1,
    deltas_map: dict[str, tuple[int, int]] | None = None,
) -> list[str]:
    """Generates directional action strings avoiding hazards."""
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

    path = safe_path(
        grid_or_frame,
        start,
        goal,
        walkable=walkable,
        fatal_colors=fatal_colors,
        threat_field=threat_field,
        step=s,
        deltas=deltas_map,
    )
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


def evasion_vector(
    avatar_pos: tuple[int, int],
    threat_positions: Iterable[tuple[int, int]],
    walkable_mask: list[list[bool]] | np.ndarray,
    deltas_map: dict[str, tuple[int, int]] | None = None,
) -> str | None:
    """Selects the best immediate action to escape nearby threats.
    
    Maximizes minimum distance to any threat position while moving to a valid walkable cell.
    """
    ar, ac = int(avatar_pos[0]), int(avatar_pos[1])
    threats = list(threat_positions)
    if not threats:
        return None

    if deltas_map:
        moves = deltas_map
    else:
        moves = {
            "ACTION1": (-1, 0),
            "ACTION2": (1, 0),
            "ACTION3": (0, -1),
            "ACTION4": (0, 1),
        }

    rows = len(walkable_mask)
    cols = len(walkable_mask[0]) if rows > 0 else 0

    best_action = None
    best_min_dist = -1.0

    for act_name, (dr, dc) in moves.items():
        nr, nc = ar + dr, ac + dc
        if 0 <= nr < rows and 0 <= nc < cols and walkable_mask[nr][nc]:
            min_dist = min(abs(nr - tr) + abs(nc - tc) for tr, tc in threats)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_action = act_name

    return best_action


def is_safe_action(
    grid_or_frame,
    avatar_pos: tuple[int, int],
    action: str,
    deltas_map: dict[str, tuple[int, int]],
    fatal_colors: Iterable[int] | None = None,
    threat_field: list[list[float]] | None = None,
) -> bool:
    """Verifies that an intended action does not step into a fatal cell or infinite hazard."""
    delta = deltas_map.get(action)
    if not delta:
        return True

    grid = to_grid(grid_or_frame)
    if not grid or len(grid) == 0:
        return False

    rows, cols = len(grid), max(len(r) for r in grid)
    nr = avatar_pos[0] + delta[0]
    nc = avatar_pos[1] + delta[1]

    if not (0 <= nr < rows and 0 <= nc < cols):
        return False

    val = grid[nr][nc] if nc < len(grid[nr]) else 0
    if fatal_colors and val in set(fatal_colors):
        return False

    if threat_field and threat_field[nr][nc] == float("inf"):
        return False

    return True

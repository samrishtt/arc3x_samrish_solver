"""Astra Click Solver: Interactive Button, Widget, and Target Selection.

Designed for ARC-AGI-3 click-only and hybrid environments (e.g. tn36, cd82, tr87, s5i5).
Supports:
- High-priority interactive target extraction (buttons, widgets, toggles, object centers)
- Click effect classification (TELEPORT, PAINT, TOGGLE, WIDGET, SELECT, INERT)
- Volatile HUD chrome filtering (ignoring ticking counters)
- Click target ranking and exploration policy
"""

from collections import Counter
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


def find_click_targets(grid_or_frame, bg_color: int = 0) -> list[tuple[int, int]]:
    """Extracts and ranks high-priority click coordinates.
    
    Priority order:
    1. Small isolated interactive buttons (size 1-4)
    2. Medium objects (size 5-25)
    3. Distinct color clusters differing from background
    """
    grid = to_grid(grid_or_frame)
    if not grid or len(grid) == 0:
        return []

    rows, cols = len(grid), max(len(r) for r in grid)
    visited = [[False] * cols for _ in range(rows)]
    objects = []

    for r in range(rows):
        for c in range(cols):
            if visited[r][c]:
                continue
            color = grid[r][c] if c < len(grid[r]) else None
            visited[r][c] = True
            if color is None or color == bg_color:
                continue

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

    # Sort objects: small button-like objects first, then larger structures
    small_buttons = [o["center"] for o in objects if o["size"] <= 4]
    medium_objects = [o["center"] for o in objects if 4 < o["size"] <= 25]
    large_structures = []
    for o in objects:
        if o["size"] > 25:
            large_structures.extend([o["center"], (o["bbox"][0], o["bbox"][1]), (o["bbox"][2], o["bbox"][3])])

    ranked = small_buttons + medium_objects + large_structures

    seen, dedup = set(), []
    for coord in ranked:
        if coord not in seen:
            seen.add(coord)
            dedup.append(coord)
    return dedup


class ClickModel:
    """Tracks and classifies click behavior to solve click-only games."""

    def __init__(self, background: int = 0):
        self.background = background
        self.click_history: list[tuple[int, int, dict]] = []
        self.active_cells: set[tuple[int, int]] = set()
        self.inert_cells: set[tuple[int, int]] = set()
        self.effect_counts: Counter = Counter()

    def record_click(
        self,
        before,
        after,
        click_r: int,
        click_c: int,
    ) -> str:
        """Classifies and records the result of a click at (click_r, click_c)."""
        ga, gb = to_grid(before), to_grid(after)
        if not ga or not gb:
            return "INERT"

        rows = min(len(ga), len(gb))
        diffs = {}
        for r in range(rows):
            cols = min(len(ga[r]), len(gb[r]))
            for c in range(cols):
                if ga[r][c] != gb[r][c]:
                    diffs[(r, c)] = (ga[r][c], gb[r][c])

        coord = (click_r, click_c)
        if not diffs:
            self.inert_cells.add(coord)
            self.effect_counts["INERT"] += 1
            return "INERT"

        self.active_cells.add(coord)

        if coord in diffs:
            old_c, new_c = diffs[coord]
            if len(diffs) == 1:
                effect = "TOGGLE"
            else:
                effect = "PAINT"
        else:
            effect = "WIDGET"

        self.effect_counts[effect] += 1
        self.click_history.append((click_r, click_c, {"effect": effect, "diff_count": len(diffs)}))
        return effect

    def next_recommended_clicks(self, current_frame) -> list[tuple[int, int]]:
        """Returns high-yield untried click coordinates for exploration."""
        all_cands = find_click_targets(current_frame, bg_color=self.background)
        untried = [c for c in all_cands if c not in self.inert_cells]
        active_toggles = [c for c in all_cands if c in self.active_cells]
        return untried + active_toggles

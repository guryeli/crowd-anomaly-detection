"""
Density analyzer.

Two levels:
  1. Global density: total person count in the scene vs config.max_expected_people
  2. Regional density: the scene is split into a grid and the person count in each cell is tracked.
"""
from __future__ import annotations

from typing import Dict, List


class DensityAnalyzer:
    def __init__(
        self,
        max_expected: int,
        warn_ratio: float,
        alarm_ratio: float,
        enable_regional: bool,
        grid_rows: int,
        grid_cols: int,
        max_per_cell: int,
    ):
        self.max_expected = max(1, int(max_expected))
        self.warn_ratio = warn_ratio
        self.alarm_ratio = alarm_ratio
        self.enable_regional = enable_regional
        self.grid_rows = max(1, int(grid_rows))
        self.grid_cols = max(1, int(grid_cols))
        self.max_per_cell = int(max_per_cell)

    def analyze(self, person_tracks: List[dict], frame_w: int, frame_h: int) -> Dict:
        total = len(person_tracks)
        ratio = total / self.max_expected

        if ratio >= self.alarm_ratio:
            global_state = "alarm"
        elif ratio >= self.warn_ratio:
            global_state = "warning"
        else:
            global_state = "normal"

        result: Dict = {
            "total_people": total,
            "max_expected": self.max_expected,
            "ratio": ratio,
            "global_state": global_state,
            "regional": None,
        }

        if self.enable_regional and total > 0:
            cell_w = frame_w / self.grid_cols
            cell_h = frame_h / self.grid_rows
            counts = [[0] * self.grid_cols for _ in range(self.grid_rows)]
            for p in person_tracks:
                cx, cy = p["center"]
                col = min(max(int(cx / cell_w), 0), self.grid_cols - 1)
                row = min(max(int(cy / cell_h), 0), self.grid_rows - 1)
                counts[row][col] += 1

            hotspots = []
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    if counts[r][c] >= self.max_per_cell:
                        hotspots.append((r, c, counts[r][c]))

            result["regional"] = {
                "counts": counts,
                "cell_w": cell_w,
                "cell_h": cell_h,
                "hotspots": hotspots,
            }

        return result

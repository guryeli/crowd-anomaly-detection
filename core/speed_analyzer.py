"""
Per-track-ID speed calculation.

For each track, derives a pixels/second speed from the center samples of the last N frames.
A single-frame delta is noisy, so a sliding-window average is used.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional


class SpeedAnalyzer:
    def __init__(self, window_frames: int, fps: float):
        self.window = max(1, window_frames)
        self.fps = max(fps, 1.0)
        self.history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.window + 1)
        )

    def update(self, frame_idx: int, person_tracks: List[dict]) -> Dict[int, float]:
        """Returns: {track_id: speed_pixels_per_sec}"""
        active_ids = set()
        speeds: Dict[int, float] = {}

        for p in person_tracks:
            tid = p["id"]
            cx, cy = p["center"]
            self.history[tid].append((frame_idx, cx, cy))
            active_ids.add(tid)

            hist = self.history[tid]
            if len(hist) >= 2:
                f0, x0, y0 = hist[0]
                f1, x1, y1 = hist[-1]
                dframes = max(f1 - f0, 1)
                dist = math.hypot(x1 - x0, y1 - y0)
                speeds[tid] = (dist / dframes) * self.fps
            else:
                speeds[tid] = 0.0

        # Drop old inactive tracks (prevent a memory leak).
        stale = [
            tid
            for tid in self.history
            if tid not in active_ids
            and frame_idx - self.history[tid][-1][0] > self.window * 3
        ]
        for tid in stale:
            del self.history[tid]

        return speeds

    @staticmethod
    def median(values: List[float]) -> Optional[float]:
        if not values:
            return None
        s = sorted(values)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

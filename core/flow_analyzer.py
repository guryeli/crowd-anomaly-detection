"""
Collective motion analysis based on optical flow (the core of Method 2).

With Farneback dense optical flow, on every frame:
  - Split the scene into a grid
  - Compute the mean magnitude in each cell
  - Build a direction histogram (8 directions)
  - Direction entropy: low entropy = coherent flight (panic signal)

Output signals:
  - mean_magnitude: mean motion strength over the whole scene
  - high_motion_cell_ratio: fraction of high-motion cells (for panic)
  - mean_direction_entropy: mean direction entropy (for flight coherence)
  - is_panic / is_sudden_motion: combined threshold decisions

Baseline learning:
  - The first N frames (config.flow_baseline_frames) are assumed to be a calm scene.
  - mean_magnitude is averaged over this window → a "normal motion" reference.
  - On later frames, if the instantaneous mean_magnitude exceeds `multiplier`
    times this reference, a "sudden motion jump" signal fires.
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np


class FlowAnalyzer:
    def __init__(
        self,
        grid_rows: int,
        grid_cols: int,
        resize_width: int,
        farneback_params: dict,
        high_magnitude: float,
        panic_cell_ratio: float,
        low_entropy_thresh: float,
        baseline_frames: int,
        sudden_motion_multiplier: float = 2.5,
    ):
        self.grid_rows = max(1, int(grid_rows))
        self.grid_cols = max(1, int(grid_cols))
        self.resize_width = int(resize_width)
        self.fb = farneback_params
        self.high_mag = float(high_magnitude)
        self.panic_cell_ratio = float(panic_cell_ratio)
        self.low_entropy_thresh = float(low_entropy_thresh)
        self.sudden_motion_multiplier = float(sudden_motion_multiplier)

        self.prev_gray: np.ndarray | None = None
        self.scale_factor: float = 1.0

        self.baseline_frames = int(baseline_frames)
        self.baseline_history: deque = deque(maxlen=self.baseline_frames)
        self.frame_count: int = 0
        self.n_bins = 8

    # ------------------------------------------------------------------
    def _resize(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape
        if w <= self.resize_width:
            self.scale_factor = 1.0
            return gray
        self.scale_factor = self.resize_width / w
        new_h = int(h * self.scale_factor)
        return cv2.resize(gray, (self.resize_width, new_h))

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_small = self._resize(gray)

        if self.prev_gray is None or self.prev_gray.shape != gray_small.shape:
            self.prev_gray = gray_small
            return self._empty_result()

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray_small,
            None,
            self.fb["pyr_scale"],
            self.fb["levels"],
            self.fb["winsize"],
            self.fb["iterations"],
            self.fb["poly_n"],
            self.fb["poly_sigma"],
            0,
        )
        self.prev_gray = gray_small

        fx = flow[..., 0]
        fy = flow[..., 1]
        magnitude = np.sqrt(fx * fx + fy * fy)
        angle = np.arctan2(fy, fx) + math.pi  # [0, 2π]

        h, w = magnitude.shape
        cell_h = h // self.grid_rows
        cell_w = w // self.grid_cols

        cell_mags = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        cell_entropies = np.full(
            (self.grid_rows, self.grid_cols), math.log(self.n_bins), dtype=np.float32
        )

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y0 = r * cell_h
                y1 = (r + 1) * cell_h if r < self.grid_rows - 1 else h
                x0 = c * cell_w
                x1 = (c + 1) * cell_w if c < self.grid_cols - 1 else w

                cell_mag = magnitude[y0:y1, x0:x1]
                cell_ang = angle[y0:y1, x0:x1]

                # Only pixels with meaningful motion (noise filter)
                mask = cell_mag > 0.5
                if mask.sum() < 20:
                    cell_mags[r, c] = float(cell_mag.mean())
                    continue

                cell_mags[r, c] = float(cell_mag[mask].mean())

                bin_indices = np.minimum(
                    (cell_ang[mask] / (2 * math.pi) * self.n_bins).astype(int),
                    self.n_bins - 1,
                )
                hist = np.bincount(
                    bin_indices, weights=cell_mag[mask], minlength=self.n_bins
                ).astype(np.float64)
                total = hist.sum()
                if total > 1e-6:
                    p = hist / total
                    nz = p[p > 0]
                    cell_entropies[r, c] = float(-(nz * np.log(nz)).sum())

        mean_mag = float(cell_mags.mean())
        high_motion_cells = int((cell_mags > self.high_mag).sum())
        total_cells = self.grid_rows * self.grid_cols
        high_motion_ratio = high_motion_cells / total_cells

        # Mean entropy of active (moving) cells only
        active_mask = cell_mags > self.high_mag * 0.5
        mean_entropy = (
            float(cell_entropies[active_mask].mean())
            if active_mask.any()
            else math.log(self.n_bins)
        )

        self.frame_count += 1
        is_baseline_learning = self.frame_count <= self.baseline_frames
        if is_baseline_learning:
            self.baseline_history.append(mean_mag)

        baseline_mag = (
            float(np.mean(self.baseline_history)) if self.baseline_history else 0.0
        )

        is_panic = (
            high_motion_ratio >= self.panic_cell_ratio
            and mean_entropy < self.low_entropy_thresh
        )

        is_sudden_motion = (
            baseline_mag > 0.1
            and mean_mag > baseline_mag * self.sudden_motion_multiplier
            and high_motion_ratio >= self.panic_cell_ratio * 0.5
        )

        return {
            "mean_magnitude": mean_mag,
            "baseline_magnitude": baseline_mag,
            "high_motion_cells": high_motion_cells,
            "high_motion_ratio": high_motion_ratio,
            "mean_direction_entropy": mean_entropy,
            "is_panic": is_panic,
            "is_sudden_motion": is_sudden_motion,
            "cell_mags": cell_mags,
            "cell_entropies": cell_entropies,
            "scale_factor": self.scale_factor,
            "is_baseline_learning": is_baseline_learning,
        }

    def _empty_result(self) -> dict:
        return {
            "mean_magnitude": 0.0,
            "baseline_magnitude": 0.0,
            "high_motion_cells": 0,
            "high_motion_ratio": 0.0,
            "mean_direction_entropy": math.log(self.n_bins),
            "is_panic": False,
            "is_sudden_motion": False,
            "cell_mags": None,
            "cell_entropies": None,
            "scale_factor": self.scale_factor,
            "is_baseline_learning": True,
        }

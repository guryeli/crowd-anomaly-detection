"""
Method 2: Optical Flow + Histogram-Based Anomaly

We do not track people. We measure collective pixel motion in the scene.
It still works in very dense crowds (when the tracker cannot separate individuals).

Signals:
  - flow_panic: high magnitude + coherent direction (flight)
  - flow_sudden_motion: a sudden motion jump relative to the baseline

Frame-skip: if cfg.frame_skip > 0, every (N+1)th frame is processed. Skipped
frames return the last annotation and metrics again.
"""
from __future__ import annotations

import time
from typing import Tuple

from core.anomaly_logic import AnomalyStateMachine
from core.flow_analyzer import FlowAnalyzer
from core.visualizer import draw_method2


class Method2Pipeline:
    name = "optical_flow_collective"

    def __init__(self, config):
        self.cfg = config
        self.flow = FlowAnalyzer(
            grid_rows=config.flow_grid_rows,
            grid_cols=config.flow_grid_cols,
            resize_width=config.flow_resize_width,
            farneback_params={
                "pyr_scale": config.farneback_pyr_scale,
                "levels": config.farneback_levels,
                "winsize": config.farneback_winsize,
                "iterations": config.farneback_iterations,
                "poly_n": config.farneback_poly_n,
                "poly_sigma": config.farneback_poly_sigma,
            },
            high_magnitude=config.flow_high_magnitude,
            panic_cell_ratio=config.flow_panic_cell_ratio,
            low_entropy_thresh=config.flow_low_entropy_thresh,
            baseline_frames=config.flow_baseline_frames,
            sudden_motion_multiplier=getattr(
                config, "flow_sudden_motion_multiplier", 2.5
            ),
        )
        self.alarm = AnomalyStateMachine(
            persistence_seconds=config.alarm_persistence_seconds,
            cooldown_seconds=config.alarm_cooldown_seconds,
            fps=config.target_fps,
        )
        self.frame_idx = 0
        self._frame_skip = max(0, int(getattr(config, "frame_skip", 0)))
        self._skip_counter = 0
        self._last_annotated = None
        self._last_metrics: dict = {}

    def process_frame(self, frame) -> Tuple:
        self.frame_idx += 1

        if self._frame_skip > 0 and self._skip_counter < self._frame_skip:
            self._skip_counter += 1
            if self._last_annotated is not None:
                metrics = dict(self._last_metrics)
                metrics["frame_idx"] = self.frame_idx
                metrics["skipped"] = True
                return self._last_annotated, metrics
        self._skip_counter = 0

        t0 = time.time()
        flow_result = self.flow.analyze(frame)

        raw_signals = {
            "flow_panic": flow_result["is_panic"]
            and not flow_result["is_baseline_learning"],
            "flow_sudden_motion": flow_result["is_sudden_motion"]
            and not flow_result["is_baseline_learning"],
        }

        anomaly_status = self.alarm.update(self.frame_idx, raw_signals)

        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        annotated = draw_method2(
            frame.copy(),
            flow_result,
            anomaly_status,
            self.frame_idx,
            fps_text=f"FPS: {fps:.1f}",
        )

        metrics = {
            "frame_idx": self.frame_idx,
            "mean_magnitude": flow_result["mean_magnitude"],
            "baseline_magnitude": flow_result["baseline_magnitude"],
            "high_motion_ratio": flow_result["high_motion_ratio"],
            "direction_entropy": flow_result["mean_direction_entropy"],
            "is_baseline_learning": flow_result["is_baseline_learning"],
            "raw_signals": raw_signals,
            "alarms": anomaly_status,
            "inference_ms": dt * 1000.0,
            "skipped": False,
        }
        self._last_annotated = annotated
        self._last_metrics = metrics
        return annotated, metrics

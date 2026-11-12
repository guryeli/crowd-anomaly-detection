"""
Method 1: YOLO + ByteTrack + Track-Level Statistics
From a collective perspective:
  - What fraction of people in the scene are "running" (panic_running_ratio)
  - Total person count (global density)
  - Whether a hotspot exists on the regional grid (regional density)

Frame-skip support: if cfg.frame_skip > 0, every (N+1)th frame is processed
(e.g. 1 = every other frame). Skipped frames return the last annotation and
metrics again (keeps the latency benchmark consistent).
"""
from __future__ import annotations

import time
from typing import Tuple

from core.anomaly_logic import AnomalyStateMachine
from core.density_analyzer import DensityAnalyzer
from core.speed_analyzer import SpeedAnalyzer
from core.visualizer import draw_method1
from detectors.person_tracker import PersonTracker


class Method1Pipeline:
    name = "tracking_collective"

    def __init__(self, config):
        self.cfg = config
        self.tracker = PersonTracker(
            model_path=config.yolo_model,
            device=config.device,
            conf=config.conf_threshold,
            iou=config.iou_threshold,
            person_id=config.person_class_id,
            tracker=config.tracker_type,
            img_size=config.img_size,
            max_det=getattr(config, "max_det", 300),
        )
        self.speed = SpeedAnalyzer(
            window_frames=config.speed_window_frames,
            fps=config.target_fps,
        )
        self.density = DensityAnalyzer(
            max_expected=config.max_expected_people,
            warn_ratio=config.density_warning_ratio,
            alarm_ratio=config.density_alarm_ratio,
            enable_regional=config.enable_regional_density,
            grid_rows=config.grid_rows,
            grid_cols=config.grid_cols,
            max_per_cell=config.max_people_per_cell,
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
        h, w = frame.shape[:2]

        person_tracks = self.tracker.track(frame)
        speeds = self.speed.update(self.frame_idx, person_tracks)
        density_info = self.density.analyze(person_tracks, w, h)

        runners = sum(
            1
            for s in speeds.values()
            if s >= self.cfg.running_speed_pixels_per_sec
        )
        total = density_info["total_people"]
        running_ratio = runners / total if total > 0 else 0.0

        is_collective_panic = (
            total >= self.cfg.panic_min_total_people
            and runners >= self.cfg.panic_min_runners
            and running_ratio >= self.cfg.panic_running_ratio
        )

        is_global_density = density_info["global_state"] == "alarm"
        is_regional_density = (
            density_info.get("regional") is not None
            and len(density_info["regional"]["hotspots"]) > 0
        )

        raw_signals = {
            "collective_panic": is_collective_panic,
            "global_density": is_global_density,
            "regional_density": is_regional_density,
        }

        anomaly_status = self.alarm.update(self.frame_idx, raw_signals)

        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        annotated = draw_method1(
            frame.copy(),
            person_tracks,
            speeds,
            density_info,
            anomaly_status,
            running_speed_thresh=self.cfg.running_speed_pixels_per_sec,
            frame_idx=self.frame_idx,
            fps_text=f"FPS: {fps:.1f}",
        )

        metrics = {
            "frame_idx": self.frame_idx,
            "total_people": total,
            "runners": runners,
            "running_ratio": running_ratio,
            "global_state": density_info["global_state"],
            "regional_hotspots": (
                len(density_info["regional"]["hotspots"])
                if density_info.get("regional")
                else 0
            ),
            "raw_signals": raw_signals,
            "alarms": anomaly_status,
            "inference_ms": dt * 1000.0,
            "skipped": False,
        }
        self._last_annotated = annotated
        self._last_metrics = metrics
        return annotated, metrics

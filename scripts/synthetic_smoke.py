"""
Builds a synthetic test video and smoke-tests both methods.

Contents:
  - 90 frames (3 s @30fps), 640x480, walking "boxes" in the scene + one runner
  - Imports the Method 1 and Method 2 pipelines and runs the frames through them

This script checks import paths, whether config is wired correctly, whether
YOLO+ByteTrack works with MPS/CUDA/CPU, and whether Farneback flow runs,
without needing a REAL surveillance video.

Usage:
    python scripts/synthetic_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from methods.method1_tracking import Method1Pipeline  # noqa: E402
from methods.method2_flow import Method2Pipeline  # noqa: E402
from utils.device import describe_device, select_device  # noqa: E402
from utils.seed import set_global_seed  # noqa: E402


def make_frame(t: int, w: int = 640, h: int = 480) -> np.ndarray:
    """A synthetic scene: gray background + 3 walking + 1 running rectangle.

    YOLO may not detect these small colored boxes as "person"; the point of
    the synthetic test is to confirm that the pipeline runs without crashing
    and that the flow analysis produces numeric signals.
    """
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    # Walkers
    for i, (cx0, cy, color) in enumerate(
        [(120, 240, (0, 200, 0)), (300, 260, (200, 100, 0)), (480, 230, (0, 150, 200))]
    ):
        cx = (cx0 + t * 2) % w
        cv2.rectangle(img, (cx - 15, cy - 30), (cx + 15, cy + 30), color, -1)
    # Runner (4x speed)
    cx = (50 + t * 8) % w
    cv2.rectangle(img, (cx - 18, 350), (cx + 18, 420), (0, 0, 255), -1)
    # Light texture (for flow)
    img[:, :, 0] = (img[:, :, 0] + (t % 30)).astype(np.uint8)
    return img


def main():
    set_global_seed(42)
    cfg = Config()
    cfg.device = select_device("auto")
    cfg.target_fps = 30.0
    cfg.headless = True
    cfg.show_window = False
    cfg.flow_baseline_frames = 15
    cfg.alarm_persistence_seconds = 0.3  # smoke test, short
    cfg.alarm_cooldown_seconds = 1.0

    print(f"[INFO] device={describe_device(cfg.device)}")
    print(f"[INFO] yolo={cfg.yolo_model} (downloaded if missing)")

    print("\n=== Method1 (tracking) smoke ===")
    m1 = Method1Pipeline(cfg)
    last_metrics = None
    for t in range(60):
        frame = make_frame(t)
        _, metrics = m1.process_frame(frame)
        last_metrics = metrics
    print(f"  last frame metrics:\n  {last_metrics}")
    assert "alarms" in last_metrics, "Method1 alarms key missing"

    print("\n=== Method2 (optical flow) smoke ===")
    m2 = Method2Pipeline(cfg)
    last_metrics = None
    for t in range(60):
        frame = make_frame(t)
        _, metrics = m2.process_frame(frame)
        last_metrics = metrics
    print(f"  last frame metrics:\n  {last_metrics}")
    assert "alarms" in last_metrics, "Method2 alarms key missing"
    assert (
        last_metrics["mean_magnitude"] > 0.0
    ), "Flow magnitude is 0 — the analyzer saw no motion"

    print("\n[OK] Smoke test passed — pipelines are running.")


if __name__ == "__main__":
    main()

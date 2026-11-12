"""
Anomaly detection from a live source (webcam or video).

Usage:
    # Method 1 (tracking)
    python live_anomaly.py --method 1 --source 0

    # Method 2 (optical flow)
    python live_anomaly.py --method 2 --source video.mp4

    # Run both side by side
    python live_anomaly.py --method both --source video.mp4

    # Change thresholds at runtime
    python live_anomaly.py --method 1 \
        --max-people 1500 --running-speed 250 \
        --panic-ratio 0.25 --panic-min-runners 8

    # Headless (server) environment
    python live_anomaly.py --method 1 --source video.mp4 --no-display --save

    # Frame skipping for performance (every other frame is processed)
    python live_anomaly.py --method 1 --source video.mp4 --frame-skip 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from methods.method1_tracking import Method1Pipeline  # noqa: E402
from methods.method2_flow import Method2Pipeline  # noqa: E402
from utils.device import describe_device, select_device  # noqa: E402
from utils.seed import set_global_seed  # noqa: E402


def stack_side_by_side(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    target_h = min(h1, h2)
    if h1 != target_h:
        img1 = cv2.resize(img1, (int(w1 * target_h / h1), target_h))
    if h2 != target_h:
        img2 = cv2.resize(img2, (int(w2 * target_h / h2), target_h))
    return np.hstack([img1, img2])


def parse_args():
    p = argparse.ArgumentParser(
        description="Crowd Anomaly Detection (live / video). Method 1 = tracking, "
        "Method 2 = optical flow."
    )
    p.add_argument("--method", choices=["1", "2", "both"], default="1")
    p.add_argument("--source", default="0", help="webcam index (0,1,...) or video path")
    p.add_argument("--device", default="auto", help="auto / cpu / mps / cuda / cuda:0")
    p.add_argument("--model", default=None, help="YOLO weights file (e.g. yolov8s.pt)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", action="store_true", help="write the output video")
    p.add_argument("--out", default=None, help="output video path (together with --save)")
    p.add_argument("--no-display", action="store_true", help="do not open a GUI window (server)")
    p.add_argument("--frame-skip", type=int, default=None,
                   help="Every (N+1)th frame is processed (for perf). 0 = no skipping")

    # Density
    p.add_argument("--max-people", type=int, default=None,
                   help="Expected upper density (global density alarm threshold)")
    p.add_argument("--max-per-cell", type=int, default=None,
                   help="Max people in one cell (regional density)")
    p.add_argument("--grid-rows", type=int, default=None)
    p.add_argument("--grid-cols", type=int, default=None)

    # Speed / panic
    p.add_argument("--running-speed", type=float, default=None,
                   help="Running threshold in pixels/second")
    p.add_argument("--panic-ratio", type=float, default=None,
                   help="Runner ratio that triggers panic (0-1)")
    p.add_argument("--panic-min-runners", type=int, default=None,
                   help="Minimum number of runners (noise filter)")
    p.add_argument("--panic-min-total", type=int, default=None,
                   help="Minimum total number of people (meaningful sample)")

    # Alarm timing
    p.add_argument("--alarm-persistence", type=float, default=None,
                   help="Consecutive seconds required to trigger an alarm")
    p.add_argument("--alarm-cooldown", type=float, default=None,
                   help="Wait time before the same alarm type can re-fire")

    # Detection
    p.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=None, help="YOLO IoU threshold")
    p.add_argument("--imgsz", type=int, default=None, help="YOLO inference image size")

    # Optical flow
    p.add_argument("--flow-high-mag", type=float, default=None)
    p.add_argument("--flow-panic-cell-ratio", type=float, default=None)
    p.add_argument("--flow-low-entropy", type=float, default=None)
    return p.parse_args()


def apply_overrides(cfg: Config, args) -> Config:
    if args.model:
        cfg.yolo_model = args.model
    if args.max_people is not None:
        cfg.max_expected_people = args.max_people
    if args.max_per_cell is not None:
        cfg.max_people_per_cell = args.max_per_cell
    if args.grid_rows is not None:
        cfg.grid_rows = args.grid_rows
    if args.grid_cols is not None:
        cfg.grid_cols = args.grid_cols
    if args.running_speed is not None:
        cfg.running_speed_pixels_per_sec = args.running_speed
    if args.panic_ratio is not None:
        cfg.panic_running_ratio = args.panic_ratio
    if args.panic_min_runners is not None:
        cfg.panic_min_runners = args.panic_min_runners
    if args.panic_min_total is not None:
        cfg.panic_min_total_people = args.panic_min_total
    if args.alarm_persistence is not None:
        cfg.alarm_persistence_seconds = args.alarm_persistence
    if args.alarm_cooldown is not None:
        cfg.alarm_cooldown_seconds = args.alarm_cooldown
    if args.conf is not None:
        cfg.conf_threshold = args.conf
    if args.iou is not None:
        cfg.iou_threshold = args.iou
    if args.imgsz is not None:
        cfg.img_size = args.imgsz
    if args.flow_high_mag is not None:
        cfg.flow_high_magnitude = args.flow_high_mag
    if args.flow_panic_cell_ratio is not None:
        cfg.flow_panic_cell_ratio = args.flow_panic_cell_ratio
    if args.flow_low_entropy is not None:
        cfg.flow_low_entropy_thresh = args.flow_low_entropy
    if args.frame_skip is not None:
        cfg.frame_skip = args.frame_skip
    if args.no_display:
        cfg.headless = True
        cfg.show_window = False
    if args.save and args.out:
        cfg.output_path = args.out
    cfg.seed = args.seed
    return cfg


def main():
    args = parse_args()
    set_global_seed(args.seed)

    cfg = Config()
    cfg.device = select_device(args.device)
    cfg = apply_overrides(cfg, args)

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {src}", file=sys.stderr)
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cfg.target_fps = src_fps if src_fps > 0 else 25.0

    print(f"[INFO] Source={src} FPS={cfg.target_fps:.1f}")
    print(f"[INFO] Device={describe_device(cfg.device)}")
    print(f"[INFO] Method={args.method} Model={cfg.yolo_model} Seed={cfg.seed}")
    print(
        f"[INFO] Tracking thresholds: max_people={cfg.max_expected_people} "
        f"running_speed={cfg.running_speed_pixels_per_sec}px/s "
        f"panic_ratio={cfg.panic_running_ratio} "
        f"panic_min_runners={cfg.panic_min_runners} "
        f"panic_min_total={cfg.panic_min_total_people} "
        f"max_per_cell={cfg.max_people_per_cell}"
    )
    print(
        f"[INFO] Flow thresholds: high_mag={cfg.flow_high_magnitude} "
        f"panic_cell_ratio={cfg.flow_panic_cell_ratio} "
        f"low_entropy={cfg.flow_low_entropy_thresh}"
    )
    print(
        f"[INFO] Alarm: persistence={cfg.alarm_persistence_seconds}s "
        f"cooldown={cfg.alarm_cooldown_seconds}s frame_skip={cfg.frame_skip}"
    )

    pipelines = []
    if args.method in ("1", "both"):
        pipelines.append(("Method1-Tracking", Method1Pipeline(cfg)))
    if args.method in ("2", "both"):
        pipelines.append(("Method2-OpticalFlow", Method2Pipeline(cfg)))

    writer = None
    win_name = "Crowd Anomaly Detection"

    n_frames = 0
    n_alarms = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            outputs = []
            for name, pipe in pipelines:
                annotated, metrics = pipe.process_frame(frame.copy())
                cv2.putText(
                    annotated,
                    f"[{name}]",
                    (10, annotated.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )
                outputs.append(annotated)

                firing = [k for k, v in metrics["alarms"].items() if v == "firing"]
                if firing and not metrics.get("skipped"):
                    n_alarms += 1
                    print(
                        f"[ALARM] frame={pipe.frame_idx} method={name} "
                        f"types={firing}"
                    )

            display = (
                outputs[0] if len(outputs) == 1 else stack_side_by_side(*outputs)
            )

            if args.save:
                if writer is None:
                    h, w = display.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        cfg.output_path, fourcc, cfg.target_fps or 25.0, (w, h)
                    )
                writer.write(display)

            if not cfg.headless:
                cv2.imshow(win_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            n_frames += 1
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, shutting down...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Output written: {cfg.output_path}")
        cv2.destroyAllWindows()
        print(f"[INFO] Processed {n_frames} frames, {n_alarms} alarm signals.")


if __name__ == "__main__":
    main()

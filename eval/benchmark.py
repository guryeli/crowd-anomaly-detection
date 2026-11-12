"""
Crowd anomaly benchmark.

Folder layout:
    data/videos/<video.(mp4|avi|...)>
    data/ground_truth.csv
        video,anomaly_start_sec,anomaly_end_sec,anomaly_type
        umn_scenario1.mp4,12.4,20.5,panic
        normal_walk.mp4,,,
    (empty anomaly_start = video with no anomaly, negative sample)

Runs both methods on every video and writes frame-level alarm times to CSV.
Comparison metrics:
  - TP / FP / FN / TN
  - Precision / Recall / F1
  - Avg / median latency (alarm delay relative to ground truth, seconds)
  - Avg / median inference ms
  - Realtime FPS

Usage:
    python -m eval.benchmark \
        --videos data/videos \
        --gt data/ground_truth.csv \
        --methods 1,2 \
        --device auto \
        --out-dir benchmark_results

Outputs:
    benchmark_results/per_video.csv     # per video × method detail
    benchmark_results/summary.csv       # aggregate table per method
    benchmark_results/anomaly_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from methods.method1_tracking import Method1Pipeline  # noqa: E402
from methods.method2_flow import Method2Pipeline  # noqa: E402
from utils.device import describe_device, select_device  # noqa: E402
from utils.seed import set_global_seed  # noqa: E402


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def run_video(video_path: str, cfg: Config, method_cls) -> Optional[Dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cfg.target_fps = src_fps

    pipeline = method_cls(cfg)
    first_alarm_frame: Optional[int] = None
    first_alarm_type: Optional[str] = None
    total_frames = 0
    inf_times: List[float] = []
    alarm_frames: List[int] = []

    t_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        _, metrics = pipeline.process_frame(frame)
        firing = [k for k, v in metrics["alarms"].items() if v == "firing"]
        if firing and first_alarm_frame is None:
            first_alarm_frame = pipeline.frame_idx
            first_alarm_type = firing[0]
        if firing:
            alarm_frames.append(pipeline.frame_idx)
        if "inference_ms" in metrics and not metrics.get("skipped"):
            inf_times.append(metrics["inference_ms"])

    wall = time.time() - t_start
    cap.release()

    return {
        "video": Path(video_path).name,
        "fps_source": src_fps,
        "total_frames": total_frames,
        "duration_sec": total_frames / src_fps if src_fps else 0.0,
        "first_alarm_frame": first_alarm_frame,
        "first_alarm_sec": first_alarm_frame / src_fps
        if first_alarm_frame and src_fps
        else None,
        "first_alarm_type": first_alarm_type,
        "any_alarm": first_alarm_frame is not None,
        "n_alarm_frames": len(alarm_frames),
        "wall_time_sec": wall,
        "realtime_fps": total_frames / wall if wall > 0 else 0.0,
        "median_inference_ms": statistics.median(inf_times) if inf_times else None,
        "mean_inference_ms": statistics.mean(inf_times) if inf_times else None,
    }


def evaluate(results: List[Dict], gt_df: pd.DataFrame, method_name: str):
    rows = []
    tp = fp = fn = tn = 0
    latencies: List[float] = []

    for r in results:
        gt_row = gt_df[gt_df["video"] == r["video"]]
        has_gt = len(gt_row) > 0 and pd.notna(gt_row["anomaly_start_sec"].values[0])
        gt_start = (
            float(gt_row["anomaly_start_sec"].values[0]) if has_gt else None
        )

        predicted = r["any_alarm"]
        if has_gt and predicted:
            tp += 1
            if r["first_alarm_sec"] is not None:
                latencies.append(r["first_alarm_sec"] - gt_start)
            label = "TP"
        elif has_gt and not predicted:
            fn += 1
            label = "FN"
        elif not has_gt and predicted:
            fp += 1
            label = "FP"
        else:
            tn += 1
            label = "TN"

        rows.append(
            {
                **r,
                "method": method_name,
                "gt_start_sec": gt_start,
                "label": label,
                "latency_sec": (
                    r["first_alarm_sec"] - gt_start
                    if r["first_alarm_sec"] is not None and gt_start is not None
                    else None
                ),
            }
        )

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
    avg_lat = statistics.mean(latencies) if latencies else None
    med_lat = statistics.median(latencies) if latencies else None

    summary = {
        "method": method_name,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": round(p, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "AvgLatency_sec": round(avg_lat, 3) if avg_lat is not None else None,
        "MedianLatency_sec": round(med_lat, 3) if med_lat is not None else None,
        "videos_evaluated": len(results),
    }
    return rows, summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True, help="video folder")
    p.add_argument("--gt", required=True, help="path to ground_truth.csv")
    p.add_argument(
        "--out-dir",
        default="benchmark_results",
        help="output folder (CSV + JSON)",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--methods", default="1,2", help="comma-separated: 1,2 / 1 / 2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frame-skip", type=int, default=0)
    p.add_argument("--model", default=None, help="YOLO weights (yolov8n.pt, etc.)")
    p.add_argument("--conf", type=float, default=None, help="YOLO confidence threshold")
    p.add_argument("--iou", type=float, default=None, help="YOLO IoU threshold")
    p.add_argument("--imgsz", type=int, default=None, help="YOLO inference image size")
    p.add_argument("--max-people", type=int, default=None)
    p.add_argument("--max-per-cell", type=int, default=None)
    p.add_argument("--grid-rows", type=int, default=None)
    p.add_argument("--grid-cols", type=int, default=None)
    p.add_argument("--running-speed", type=float, default=None)
    p.add_argument("--panic-ratio", type=float, default=None)
    p.add_argument("--panic-min-runners", type=int, default=None)
    p.add_argument("--panic-min-total", type=int, default=None)
    p.add_argument("--alarm-persistence", type=float, default=None)
    p.add_argument("--alarm-cooldown", type=float, default=None)
    p.add_argument("--flow-high-mag", type=float, default=None)
    p.add_argument("--flow-panic-cell-ratio", type=float, default=None)
    p.add_argument("--flow-low-entropy", type=float, default=None)
    return p.parse_args()


def apply_overrides(cfg: Config, args) -> Config:
    if args.model:
        cfg.yolo_model = args.model
    if args.conf is not None:
        cfg.conf_threshold = args.conf
    if args.iou is not None:
        cfg.iou_threshold = args.iou
    if args.imgsz is not None:
        cfg.img_size = args.imgsz
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
    if args.flow_high_mag is not None:
        cfg.flow_high_magnitude = args.flow_high_mag
    if args.flow_panic_cell_ratio is not None:
        cfg.flow_panic_cell_ratio = args.flow_panic_cell_ratio
    if args.flow_low_entropy is not None:
        cfg.flow_low_entropy_thresh = args.flow_low_entropy
    cfg.frame_skip = args.frame_skip
    cfg.headless = True
    cfg.show_window = False
    cfg.seed = args.seed
    return cfg


def main():
    args = parse_args()
    set_global_seed(args.seed)

    cfg_template = Config()
    cfg_template.device = select_device(args.device)
    cfg_template = apply_overrides(cfg_template, args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.gt):
        print(f"[WARN] GT file missing: {args.gt} → all videos counted as negative.")
        gt_df = pd.DataFrame(
            columns=["video", "anomaly_start_sec", "anomaly_end_sec", "anomaly_type"]
        )
    else:
        gt_df = pd.read_csv(args.gt, comment="#")

    video_files = sorted(
        [
            p
            for p in Path(args.videos).iterdir()
            if p.suffix.lower() in VIDEO_EXTS
        ]
    )
    if not video_files:
        print(f"[ERROR] No videos under {args.videos}.", file=sys.stderr)
        sys.exit(1)

    selected = set(s.strip() for s in args.methods.split(","))
    method_map = {
        "1": ("M1-Tracking", Method1Pipeline),
        "2": ("M2-OpticalFlow", Method2Pipeline),
    }

    print(f"[INFO] Device={describe_device(cfg_template.device)}")
    print(f"[INFO] {len(video_files)} video, methods={sorted(selected)}")
    print(f"[INFO] Frame-skip={cfg_template.frame_skip} Seed={cfg_template.seed}")

    all_rows = []
    summaries = []

    for mid in sorted(selected):
        if mid not in method_map:
            continue
        name, cls = method_map[mid]
        print(f"\n=== {name} ===")
        results = []
        for vp in video_files:
            print(f"[RUN] {vp.name}")
            cfg = Config()
            cfg.__dict__.update(cfg_template.__dict__)
            r = run_video(str(vp), cfg, cls)
            if r:
                print(
                    f"      alarm={r['any_alarm']} "
                    f"first_alarm_sec={r['first_alarm_sec']} "
                    f"type={r['first_alarm_type']} "
                    f"realtime_fps={r['realtime_fps']:.1f}"
                )
                results.append(r)
        rows, summary = evaluate(results, gt_df, name)
        all_rows.extend(rows)
        summaries.append(summary)

    per_video_csv = out_dir / "per_video.csv"
    summary_csv = out_dir / "summary.csv"
    json_path = out_dir / "anomaly_benchmark.json"

    pd.DataFrame(all_rows).to_csv(per_video_csv, index=False)
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)

    payload = {
        "schema_version": 1,
        "device": cfg_template.device,
        "seed": cfg_template.seed,
        "frame_skip": cfg_template.frame_skip,
        "model": cfg_template.yolo_model,
        "thresholds": {
            "running_speed_pixels_per_sec": cfg_template.running_speed_pixels_per_sec,
            "panic_running_ratio": cfg_template.panic_running_ratio,
            "panic_min_runners": cfg_template.panic_min_runners,
            "panic_min_total_people": cfg_template.panic_min_total_people,
            "max_expected_people": cfg_template.max_expected_people,
            "max_people_per_cell": cfg_template.max_people_per_cell,
            "flow_high_magnitude": cfg_template.flow_high_magnitude,
            "flow_panic_cell_ratio": cfg_template.flow_panic_cell_ratio,
            "flow_low_entropy_thresh": cfg_template.flow_low_entropy_thresh,
            "alarm_persistence_seconds": cfg_template.alarm_persistence_seconds,
            "alarm_cooldown_seconds": cfg_template.alarm_cooldown_seconds,
        },
        "summaries": summaries,
        "per_video": all_rows,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\n=== COMPARISON ===")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\n[OUT] per-video CSV : {per_video_csv}")
    print(f"[OUT] summary  CSV : {summary_csv}")
    print(f"[OUT] JSON         : {json_path}")
    print(
        "\nFor figures: "
        f"python -m eval.plot_results --json {json_path} --out-dir figures"
    )


if __name__ == "__main__":
    main()

"""
Crowd Anomaly Detection - central configuration.

Set every threshold here. They need to be tuned per scene/camera.
CLI arguments can override them at runtime (live_anomaly.py / eval/benchmark.py).
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---------- Detection (both methods) ----------
    yolo_model: str = "yolov8s.pt"          # fast motion / distant person: yolov8m.pt or yolov8l.pt
    device: str = "auto"                     # auto / cpu / mps / cuda
    conf_threshold: float = 0.18             # lower = higher recall under motion blur; higher FP risk
    iou_threshold: float = 0.45
    person_class_id: int = 0
    img_size: int = 960                      # YOLO inference; 640→960/1280 helps small/fast targets
    max_det: int = 300                       # upper detection count in crowded scenes

    # ---------- Tracker (Method 1) ----------
    tracker_type: str = "bytetrack.yaml"     # ultralytics built-in: bytetrack or botsort

    # ---------- Method 1: Tracking-based collective ----------
    speed_window_frames: int = 5             # sliding window for noise reduction
    pixel_per_meter: float = 50.0            # optional; meters/sec if calibrated

    # Running detection
    running_speed_pixels_per_sec: float = 200.0
    # NOTE: in a close 1080p shot, running speed is typically ~150-300 px/s.
    # Measure the scene's median walking speed and set the running threshold to 2-3x.

    # Collective panic (mass running) thresholds
    panic_running_ratio: float = 0.30        # if 30%+ of people in the scene are running
    panic_min_runners: int = 3               # at least 3 people must be running (noise filter)
    panic_min_total_people: int = 5          # at least 5 people total (meaningful sample)

    # Density anomaly
    max_expected_people: int = 500
    density_warning_ratio: float = 0.8       # yellow warning at 80% of max
    density_alarm_ratio: float = 1.0         # red alarm at 100% of max

    # Regional density (split the scene into a grid)
    enable_regional_density: bool = True
    grid_rows: int = 3
    grid_cols: int = 3
    max_people_per_cell: int = 80

    # Alarm triggering (UMN ablation: recall rose 5x with 2.0/5.0 → 1.0/3.0)
    alarm_persistence_seconds: float = 1.0   # consecutive seconds of signal → alarm
    alarm_cooldown_seconds: float = 3.0      # seconds with no re-fire after a trigger

    # ---------- Method 2: Optical flow collective ----------
    flow_grid_rows: int = 6
    flow_grid_cols: int = 8                  # 6x8 = 48 cells
    flow_resize_width: int = 640             # shrink the frame for flow (for speed)

    # Farneback parameters
    farneback_pyr_scale: float = 0.5
    farneback_levels: int = 3
    farneback_winsize: int = 15
    farneback_iterations: int = 3
    farneback_poly_n: int = 5
    farneback_poly_sigma: float = 1.2

    # Collective signals (UMN ablation: recall rose 2x with 4.0/0.15 → 3.0/0.10)
    flow_high_magnitude: float = 3.0         # pixels/frame; above this is "high motion"
    flow_panic_cell_ratio: float = 0.10      # 10%+ of cells at high mag → panic
    flow_low_entropy_thresh: float = 1.5     # max ~3.0 for 8 directions; low = coherent flight
    flow_baseline_frames: int = 60           # baseline learning duration

    # Jump threshold relative to the baseline
    flow_sudden_motion_multiplier: float = 2.5

    # ---------- Performance ----------
    frame_skip: int = 0                      # if >0, every (N+1)th frame is processed (e.g. 1 = every other frame)

    # ---------- Visualization ----------
    show_window: bool = True
    save_video: bool = False
    output_path: str = "output_anomaly.mp4"
    headless: bool = False                   # environments without a GUI (e.g. Docker, server)

    # ---------- Source ----------
    target_fps: float = 30.0                 # source FPS overwrites this automatically
    seed: int = 42

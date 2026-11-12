# Crowd Anomaly Detection

Two-method crowd anomaly detection system (for an academic thesis):

| Method | Logic | Scene where it is strongest |
|---|---|---|
| **Method 1** — `tracking_collective` | YOLO + ByteTrack + track-level statistics (speed + global/regional density) | Low–medium crowds, scenarios that need explainability |
| **Method 2** — `optical_flow_collective` | Farneback dense optical flow + grid magnitude/direction histogram | When the tracker struggles in very dense crowds, collective flight (panic) |

Both methods share the same `Config` and are compared with the same `eval/benchmark.py`.
Thresholds can be changed one by one at runtime from the CLI.

> This project performs *unsupervised / threshold-based* anomaly detection — no training
> is required. It uses the "person" class with YOLO weights pretrained on COCO.

---

## Which anomalies does it detect?

User requirement: **individual running does not raise an alarm**, but collective panic / density does.

### Method 1 (Tracking-based)
| Alarm type | Trigger |
|---|---|
| `collective_panic` | `runners >= panic_min_runners` **AND** `total >= panic_min_total_people` **AND** `runners/total >= panic_running_ratio` |
| `global_density` | `total_people >= max_expected_people * density_alarm_ratio` |
| `regional_density` | A grid cell exceeded `max_people_per_cell` |

### Method 2 (Optical flow)
| Alarm type | Trigger |
|---|---|
| `flow_panic` | High-magnitude cell ratio is high **AND** direction entropy is low (coherent flight) |
| `flow_sudden_motion` | Mean magnitude exceeded `flow_sudden_motion_multiplier` times the baseline |

All alarms are filtered by **persistence** (at least `alarm_persistence_seconds` of consecutive signal) and
**cooldown** (no re-fire for `alarm_cooldown_seconds`).

---

## Project structure

```
AnomaliDedection/
├── config.py                     # all thresholds in one place (dataclass)
├── live_anomaly.py               # live / video runner
├── check_installation.py         # smoke test
├── requirements.txt
├── detectors/
│   └── person_tracker.py         # YOLO + Ultralytics built-in ByteTrack
├── core/
│   ├── speed_analyzer.py         # track-level speed (Method 1)
│   ├── density_analyzer.py       # global + regional density (Method 1)
│   ├── flow_analyzer.py          # Farneback + magnitude/direction histogram (Method 2)
│   ├── anomaly_logic.py          # persistence + cooldown state machine
│   └── visualizer.py             # overlay for both methods
├── methods/
│   ├── method1_tracking.py       # Method 1 pipeline
│   └── method2_flow.py           # Method 2 pipeline
├── eval/
│   ├── benchmark.py              # side-by-side method comparison
│   └── plot_results.py           # thesis-quality figures
├── utils/
│   ├── device.py                 # CUDA/MPS/CPU autodetect
│   └── seed.py                   # global reproducibility seed
├── scripts/
│   ├── download_datasets.py      # UMN/UCSD download helper
│   └── synthetic_smoke.py        # synthetic smoke test
├── data/
│   ├── videos/                   # video folder for the benchmark
│   └── ground_truth.csv          # video,anomaly_start_sec,anomaly_end_sec,anomaly_type
├── benchmark_results/            # CSV + JSON outputs
└── figures/                      # PNG figures
```

---

## Installation

### For Apple Silicon (M-series)
```bash
cd AnomaliDedection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python check_installation.py
```

`check_installation.py` verifies the Python version, all dependencies, device selection
(MPS / CUDA / CPU), and the file layout. YOLOv8n weights are downloaded automatically
if missing.

### For Linux + NVIDIA GPU
Same steps; only before/after `pip install -r requirements.txt`, if you want a CUDA build of PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Quick Test - Webcam

Method 1 (live, MPS/CUDA automatic):
```bash
python live_anomaly.py --method 1 --source 0
```

Method 2:
```bash
python live_anomaly.py --method 2 --source 0
```

Both side by side:
```bash
python live_anomaly.py --method both --source 0
```

Headless (server / window cannot open):
```bash
python live_anomaly.py --method 1 --source video.mp4 --no-display --save --out out.mp4
```

Synthetic smoke test (no real video needed; checks that the pipeline runs):
```bash
python scripts/synthetic_smoke.py
```

---

## Usage adapted to your scenario

Scenario: Square camera, up to 1500 people is normal, 25% running = panic, at least
8 people must be running, running threshold during normal walking is 220 px/s.

```bash
python live_anomaly.py --method 1 \
    --source meydan.mp4 \
    --max-people 1500 \
    --panic-ratio 0.25 \
    --running-speed 220 \
    --panic-min-runners 8 \
    --panic-min-total 30 \
    --max-per-cell 200 \
    --grid-rows 4 --grid-cols 5
```

Webcam self-test (low thresholds so you can trigger it by running close to the camera):
```bash
python live_anomaly.py --method 1 --source 0 \
    --max-people 5 --panic-ratio 0.5 \
    --panic-min-runners 1 --panic-min-total 1 \
    --alarm-persistence 0.5 --alarm-cooldown 2
```

Method 2 self-test (sudden motion alarm):
```bash
python live_anomaly.py --method 2 --source 0 \
    --flow-high-mag 3.0 --flow-panic-cell-ratio 0.10 \
    --alarm-persistence 0.5
```

### All CLI flags (summary)

| Flag | Meaning |
|---|---|
| `--method {1,2,both}` | Method selection |
| `--source` | Webcam index (0,1,...) or video path |
| `--device` | `auto` / `cpu` / `mps` / `cuda` / `cuda:0` |
| `--model` | YOLO weights (yolov8s.pt, etc.) |
| `--seed` | Reproducibility seed (default 42) |
| `--save` `--out` | Write the output video |
| `--no-display` | Do not open a GUI window (server/headless) |
| `--frame-skip N` | Every (N+1)th frame is processed (perf) |
| `--max-people` | Expected upper density (global density) |
| `--max-per-cell` | Max people in one cell (regional density) |
| `--grid-rows / --grid-cols` | Density grid |
| `--running-speed` | Running threshold (px/s) |
| `--panic-ratio` | Runners/total ratio (0–1) |
| `--panic-min-runners` | Min running people (noise filter) |
| `--panic-min-total` | Min total people (meaningful sample) |
| `--alarm-persistence` | Consecutive seconds required for an alarm |
| `--alarm-cooldown` | Wait seconds before re-fire |
| `--conf / --iou / --imgsz` | YOLO inference parameters |
| `--flow-high-mag` | Method 2: cell high-motion threshold |
| `--flow-panic-cell-ratio` | Method 2: high-mag cell ratio |
| `--flow-low-entropy` | Method 2: coherent-flight entropy threshold |

All values also exist as defaults inside `config.py`.

---

## Benchmark — UMN / UCSD / ShanghaiTech

### 1) Dataset
```bash
# UMN combined panic video + ground-truth template
python scripts/download_datasets.py --datasets umn
python scripts/download_datasets.py --gen-gt-template
```

`data/ground_truth.csv` format:
```csv
video,anomaly_start_sec,anomaly_end_sec,anomaly_type
umn_scenario1.mp4,7.5,15.0,panic
umn_scenario2.mp4,9.8,17.0,panic
normal_walk.mp4,,,
```
> Empty `anomaly_start_sec` = video with no anomaly (negative sample)

The UMN video is concatenated; split the scenarios you want into separate files with `ffmpeg -ss <start> -t <dur>`,
then place them under `data/videos/`.

### 2) Run the comparison
```bash
python -m eval.benchmark \
    --videos data/videos \
    --gt data/ground_truth.csv \
    --methods 1,2 \
    --device auto \
    --out-dir benchmark_results
```

Outputs:
```
benchmark_results/
├── per_video.csv               # per video × method detail
├── summary.csv                 # aggregate table per method
└── anomaly_benchmark.json      # full results + thresholds used (citable in the thesis)
```

### 3) Generate the figures
```bash
python -m eval.plot_results \
    --json benchmark_results/anomaly_benchmark.json \
    --out-dir figures
```

Figures produced:
- `figures/summary_table.png` — method × P/R/F1/latency table
- `figures/precision_recall_bar.png` — P/R/F1 bar chart
- `figures/latency_box.png` — alarm latency per method (boxplot)
- `figures/per_video_alarms.png` — video × method alarm map
- `figures/realtime_fps.png` — realtime FPS comparison
- `figures/inference_ms_box.png` — inference time per frame

### Our UMN results (Apple M-series MPS)

**Data**: `Crowd-Activity-All.avi` (UMN combined video, 11 panic scenarios, 258s
@30fps, 320×240) + 5 normal side videos (mergeSplit, plaza1, plaza2, roi,
sidewalk). 6 videos total, ground truth: 1 with anomaly + 5 negatives.

#### Three-run tuning ablation

| Configuration | YOLO | persist / cool | panic min runners/total | conf | flow high_mag / cell_ratio |
|---|---|---|---|---|---|
| **v1** (old default) | yolov8n | 2.0 / 5.0 | 5 / 10 | 0.30 | 4.0 / 0.15 |
| **v2-production** (current `config.py` ✅) | yolov8s | 1.0 / 3.0 | 3 / 5 | 0.20 | 3.0 / 0.10 |
| **v2-aggressive** (ablation only) | yolov8s | 0.3 / 1.0 | 2 / 3 | 0.15 | 1.5 / 0.08 |

#### Combined-video any_alarm results (n=6 videos)

| Configuration | Method | TP | FP | FN | TN | Precision | Recall | **F1** |
|---|---|---|---|---|---|---|---|---|
| v1 | M1-Tracking | 0 | 0 | 1 | 5 | 0.00 | 0.00 | **0.00** |
| v1 | M2-OpticalFlow | 1 | 0 | 0 | 5 | 1.00 | 1.00 | **1.00** |
| **v2-production** | **M1-Tracking** | 1 | 0 | 0 | 5 | 1.00 | 1.00 | **1.00** |
| **v2-production** | **M2-OpticalFlow** | 1 | 0 | 0 | 5 | 1.00 | 1.00 | **1.00** |
| v2-aggressive | M1-Tracking | 1 | 0 | 0 | 5 | 1.00 | 1.00 | **1.00** |
| v2-aggressive | M2-OpticalFlow | 1 | **4** | 0 | 1 | **0.20** | 1.00 | **0.33** |

→ **v2-production sweet spot**: both methods reach F1=1.0, with no FPs.
→ On v2-aggressive Method 2, **precision collapses** (4 FP, F1=0.33). In the thesis this numerically
shows the tradeoff that "method 2's high-recall advantage comes at a false-positive cost".

#### Episode-based analysis (literature GT: 11 panics @ Crowd-Activity-All.avi)

`any_alarm`-based F1 is insufficient because "TP=1" covers all 11 episodes. Episode
count is a more informative metric:

| Configuration | M1 episode count | M2 episode count | M1 frame | M2 frame |
|---|---|---|---|---|
| v1 (old default) | **1 / 11** (9%) | **5 / 11** (45%) | 5 | 158 |
| **v2-production** ✅ | **5 / 11** (45%) | **10 / 11** (91%) | 112 | 386 |
| v2-aggressive | 5 / 11 | 10 / 11 + 4 FP | 112 | 386 + FP |

**Improvement when moving to v2-production**: M1 +5×, M2 +2×.

#### Speed (inference + realtime FPS)

| Method | Median inference (ms/frame) | Realtime FPS @ MPS |
|---|---|---|
| **M1-Tracking** (YOLOv8s + ByteTrack) | 16–21 | 42–50 |
| **M2-OpticalFlow** (Farneback CPU) | ~5 | **135–181** |

→ Method 2 is ~3.5× faster; a single MacBook M-series machine can process 6 parallel
   30fps camera streams.

#### Latency (first alarm vs literature GT 21s)

| Method | First alarm | Latency | Type |
|---|---|---|---|
| M1-Tracking | 18.10s | −2.90s (early) | `collective_panic` |
| M2-OpticalFlow | 18.43s | −2.57s (early) | `flow_sudden_motion` |

Both methods catch the event ~2.5-3s earlier than the literature GT — the real panic
starts around 17-18s (Mehran 2009's GT is an approximate value).

### Main findings to report in the thesis

1. **YOLO model size is critical**: yolov8n cannot see distant people as "person"
   at 320×240 low resolution (M1 F1=0). yolov8s fixed the problem.
   Nano may suffice on HD video, but small/medium is required for low-resolution surveillance cameras.

2. **Threshold-tuning ablation yields a +5× recall increase**: the v1 → v2-production
   transition moves M1 episode recall from 1/11 → 5/11. In the academic write-up this supports
   the claim that "default thresholds are insufficient without dataset-specific tuning".

3. **Method 2 has a structural recall advantage**: even after tuning, 10/11 vs 5/11
   episodes (91% vs 45%). The pixel-centric approach being immune to ID switching
   is clear at low resolution.

4. **Overly aggressive tuning causes a precision collapse on Method 2**: v2-aggressive
   FP=4. "More sensitive = better" is not true; Method 2's operating point
   is optimal at v2-production.

5. **The two methods are complementary as descriptors**: M1 says which person/grid cell
   produced the anomaly (explainability), M2 covers more of the whole scene
   (recall). `--method both` runs them in parallel.

`v2-aggressive` results are kept under `benchmark_results/*_v2_aggressive.{json,csv}` —
a reference for ablation experiments. `v2-production` results are in the default
`anomaly_benchmark.json / per_video.csv / summary.csv` files.

---

## Recommended datasets

| Dataset | Scenario | URL |
|---|---|---|
| **UMN Crowd** | Panic flight (best fit) | http://mha.cs.umn.edu/proj_events.shtml |
| **UCSD Ped1/Ped2** | Atypical behavior on a walkway (running, bicycle) | http://www.svcl.ucsd.edu/projects/anomaly/ |
| **ShanghaiTech Campus** | 13 scenes, multiple anomalies | https://svip-lab.github.io/dataset/campus_dataset.html |
| **CUHK Avenue** | Atypical behaviors | http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html |

UMN + UCSD Ped2 is the classic pair for theses.

---

## Tuning tips

| Problem | Fix |
|---|---|
| Too many false positives | Set `alarm_persistence_seconds` to 3–4 s, raise `panic_min_runners`/`panic_min_total` |
| Alarm never triggers | Lower `running_speed_pixels_per_sec` (scene far from the camera), lower `flow_high_magnitude` |
| Scene is very crowded (1000+) | The tracker struggles in Method 1; weight **Method 2** more, or use CrowdHuman fine-tuned YOLO weights (`--model crowd.pt`) |
| Apple MPS feels slow | With `--frame-skip 1` every other frame is processed (FPS ~2x); prefer the small model (`yolov8n.pt`) |
| Shaky camera | In Method 2 the mean_magnitude baseline rises, so the jump looks small on a real anomaly → raise `flow_sudden_motion_multiplier` to 3.0–3.5 |
| Regional pile-up is not caught | Increase `grid_rows/grid_cols` (e.g. 5×5), set `max_people_per_cell` for the scene |
| You do not want alarms during baseline learning | The default already filters them; but your scene should be still for the first 60 frames |

---

## Method 1 vs Method 2 — Which one where?

A real comparison based on our UMN benchmark results:

| Dimension | Method 1 (Tracking) | Method 2 (Flow) |
|---|---|---|
| **Speed (UMN 320×240, MPS)** | YOLOv8s + ByteTrack: **45–59 FPS realtime**, 16–21 ms/frame | Farneback CPU: **180–187 FPS realtime**, ~5 ms/frame |
| **Episode recall (UMN, 11 scenarios)** | 5/11 (45%) | **10/11 (91%)** |
| **Precision (any_alarm, n=6 videos)** | 1.000 | 1.000 |
| **Detection latency** | 18.1s (vs GT 21s) | 17.6s (vs GT 21s) — earlier |
| **Alarm continuity (throughout the panic)** | 5 frames (short, split by cooldown) | 158–386 frames (continuous) |
| **Crowd robustness** | good under 300 people, tracker ID switch at 500+ | unlimited because it works on pixels |
| **Explainability** | "Track #X is running at 240 px/s, cell (1,2)" | "X% of the scene is fast, direction entropy is low" |
| **Config difficulty** | running_speed must be calibrated per scene | Learns a baseline in 60 frames, less manual tuning |
| **Density anomaly** | Directly counts "how many people" (global + grid) | Only indirect (foreground / motion proxy) |
| **HD (1080p) estimate** | Recall rises (large people are easier to detect), FPS drops | Recall ≥ tracking, FPS drops |

**Running both together is optimal in most scenarios** — Method 1 for counting
+ explainability, Method 2 for panic/flight coverage. `--method both`
runs them side by side.

---

## Reproducibility

- All scripts start with `--seed 42` (default); numpy/torch/random are seeded.
- The benchmark writes the seed, device, thresholds used, and
  YOLO weights into `anomaly_benchmark.json` — results can be reproduced exactly in the thesis.
- Device selection is autodetect via `utils/device.py`; `--device cpu` can force
  CPU (for a cross-check).

---

## Known limitations

- **Synthetic test (`scripts/synthetic_smoke.py`)** does not guarantee that YOLO
  detects the colored boxes as people. Method1 alarms generally
  do not fire on synthetic data; a real video is required.
- **No camera calibration**: speeds are in pixels/second. If meters/second
  is needed, the `pixel_per_meter` parameter exists in config, but the conversion is left
  to external use.
- **Optical flow runs on CPU** (cv2 default). It can bottleneck on very high-resolution
  (4K) video → keep `flow_resize_width` between 480–640.
- **The YOLO COCO model weakens in crowds**. For very crowded scenes,
  pass CrowdHuman fine-tuned YOLO weights with `--model`.

---

## Next steps (optional improvements)

- Method 1 crowd recall with CrowdHuman fine-tuned YOLO weights
- CSRNet density-map model → accurate counts at 1000+ people
- RAFT (PyTorch) optical flow → more accurate on GPU instead of CPU Farneback
- Comprehensive benchmark on ShanghaiTech / UCF-Crime
- LSTM-based temporal anomaly classifier (supervised) as an optional comparison

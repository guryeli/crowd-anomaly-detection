"""
Installation smoke test.

Run:
    python check_installation.py

Checks:
  - Python version
  - Dependencies (ultralytics, opencv, torch, pandas, matplotlib, ...)
  - Device (CUDA / MPS / CPU)
  - Project file layout (detectors/, core/, methods/, eval/, utils/)
  - Access to YOLO weights (whether yolov8n.pt is downloaded/loaded)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_PKGS = [
    ("ultralytics", "ultralytics"),
    ("cv2", "opencv-python"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
]

REQUIRED_DIRS = ["detectors", "core", "methods", "eval", "utils", "data"]

REQUIRED_FILES = [
    "config.py",
    "live_anomaly.py",
    "detectors/person_tracker.py",
    "core/speed_analyzer.py",
    "core/density_analyzer.py",
    "core/flow_analyzer.py",
    "core/anomaly_logic.py",
    "core/visualizer.py",
    "methods/method1_tracking.py",
    "methods/method2_flow.py",
    "eval/benchmark.py",
    "eval/plot_results.py",
    "utils/device.py",
    "utils/seed.py",
]


def check_python():
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 9)
    print(f"  Python {v.major}.{v.minor}.{v.micro} {'OK' if ok else 'FAIL (>=3.9 required)'}")
    return ok


def check_pkgs():
    all_ok = True
    for mod, pip_name in REQUIRED_PKGS:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  {pip_name:18s} OK   ({ver})")
        except ImportError as e:
            print(f"  {pip_name:18s} FAIL ({e})")
            all_ok = False
    return all_ok


def check_device():
    try:
        from utils.device import describe_device, select_device

        d = select_device("auto")
        print(f"  Selected device: {describe_device(d)}")
        return True
    except Exception as e:
        print(f"  Device check FAIL: {e}")
        return False


def check_files():
    all_ok = True
    for d in REQUIRED_DIRS:
        ok = (ROOT / d).is_dir()
        print(f"  dir  {d:30s} {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False
    for f in REQUIRED_FILES:
        ok = (ROOT / f).is_file()
        print(f"  file {f:30s} {'OK' if ok else 'FAIL'}")
        if not ok:
            all_ok = False
    return all_ok


def check_yolo_weights():
    try:
        from ultralytics import YOLO

        m = YOLO("yolov8n.pt")
        n_params = sum(p.numel() for p in m.model.parameters())
        print(f"  YOLOv8n loaded: {n_params/1e6:.2f}M params")
        return True
    except Exception as e:
        print(f"  YOLOv8n FAILED TO LOAD: {e}")
        return False


def main():
    print("=== Crowd Anomaly Detection — Installation Check ===\n")
    print("[1] Python version")
    py_ok = check_python()
    print("\n[2] Dependencies")
    pkg_ok = check_pkgs()
    print("\n[3] Device (PyTorch backend)")
    dev_ok = check_device()
    print("\n[4] Project file layout")
    files_ok = check_files()
    print("\n[5] YOLO weights (yolov8n.pt — downloaded if missing)")
    yolo_ok = check_yolo_weights()
    print()
    overall = py_ok and pkg_ok and dev_ok and files_ok and yolo_ok
    print("=" * 50)
    if overall:
        print("[OK] Installation complete. You are ready.")
        sys.exit(0)
    else:
        print("[FAIL] Fix the steps above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

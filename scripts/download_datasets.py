"""
Dataset download helper.

Supported:
  - UMN Crowd Anomaly: panic/flight scenarios (best fit, "collective panic")
  - UCSD Anomaly Ped1/Ped2: atypical behavior on a walkway (running, bicycle)

NOTE: The official UMN/UCSD sites are sometimes unreachable.
Several mirrors are tried. Manually downloading from the official site and
copying into data/videos/ is also acceptable.

Usage:
    python scripts/download_datasets.py --datasets umn,ucsd
    python scripts/download_datasets.py --gen-gt-template
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
VIDEOS_DIR = DATA_DIR / "videos"

UMN_MIRRORS = [
    # UMN crowd dataset combined video (a single file on the official site).
    "http://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi",
]

UCSD_MIRRORS = [
    "http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz",
]

GT_TEMPLATE = """video,anomaly_start_sec,anomaly_end_sec,anomaly_type
# UMN combined video; approximate panic-start seconds for each scenario:
# (The recording contains many anomalies; split out whichever ones you
#  separate from your scene and mark them line by line here.)
# umn_scenario1.mp4,7.5,15.0,panic
# umn_scenario2.mp4,9.8,17.0,panic
# umn_scenario3.mp4,8.0,16.0,panic
# UCSD Ped2 tests (converted from frame numbers at FPS=25):
# UCSDped2_Test004.mp4,2.4,8.0,running
# normal_walk.mp4,,,
"""


def download(url: str, dst: Path) -> bool:
    print(f"[GET] {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
        size = dst.stat().st_size / 1024 / 1024
        print(f"  → {dst} ({size:.1f} MB)")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        if dst.exists():
            dst.unlink()
        return False


def get_umn() -> bool:
    out = VIDEOS_DIR / "umn_crowd_activity_all.avi"
    if out.exists():
        print(f"[SKIP] UMN already present: {out}")
        return True
    for url in UMN_MIRRORS:
        if download(url, out):
            print(
                "  NOTE: The UMN video is long and contains multiple scenarios.\n"
                "       Split it with ffmpeg if you need to:\n"
                "       ffmpeg -i umn_crowd_activity_all.avi -ss 0 -t 30 umn_scenario1.mp4"
            )
            return True
    return False


def get_ucsd() -> bool:
    out = DATA_DIR / "UCSD_Anomaly_Dataset.tar.gz"
    if out.exists():
        print(f"[SKIP] UCSD already present: {out}")
        return True
    for url in UCSD_MIRRORS:
        if download(url, out):
            print(
                "  NOTE: The tar.gz contains frame directories; example conversion to video:\n"
                "       ffmpeg -framerate 10 -i Test004/%03d.tif -c:v libx264 "
                "UCSDped2_Test004.mp4"
            )
            return True
    return False


def gen_gt_template():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    p = DATA_DIR / "ground_truth.csv"
    if p.exists():
        bak = p.with_suffix(".csv.bak")
        shutil.copy2(p, bak)
        print(f"[BACKUP] existing → {bak}")
    p.write_text(GT_TEMPLATE)
    print(f"[OK] {p} written (comment lines start with #).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        default="umn",
        help="comma-separated: umn,ucsd",
    )
    ap.add_argument(
        "--gen-gt-template",
        action="store_true",
        help="write a template for data/ground_truth.csv",
    )
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    if args.gen_gt_template:
        gen_gt_template()
        return

    selected = set(s.strip().lower() for s in args.datasets.split(","))
    failures = []
    if "umn" in selected and not get_umn():
        failures.append("umn")
    if "ucsd" in selected and not get_ucsd():
        failures.append("ucsd")

    if failures:
        print(f"\n[WARN] Failed downloads: {failures}")
        print(
            "Download manually:\n"
            "  UMN  : http://mha.cs.umn.edu/proj_events.shtml\n"
            "  UCSD : http://www.svcl.ucsd.edu/projects/anomaly/dataset.html"
        )
        sys.exit(2)
    print("\n[OK] Download complete.")


if __name__ == "__main__":
    main()

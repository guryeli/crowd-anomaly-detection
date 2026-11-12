"""
Draws track, speed, density grid, and alarm overlays on the frame.

Two separate drawing functions:
  - draw_method1: tracker-based pipeline (person bbox + grid + HUD)
  - draw_method2: optical flow pipeline (magnitude heatmap + HUD)
"""
from __future__ import annotations

import cv2

COLOR_RUNNING = (0, 0, 255)         # red bbox = running
COLOR_WALK = (0, 200, 0)            # green bbox = normal
COLOR_HOTSPOT = (0, 100, 255)       # orange cell = overcrowded
COLOR_ALARM_BANNER = (0, 0, 200)
COLOR_WARN_BANNER = (0, 165, 255)


def _put_text(img, text, org, scale=0.6, color=(255, 255, 255), thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_method1(
    frame,
    person_tracks,
    speeds,
    density_info,
    anomaly_status,
    running_speed_thresh,
    frame_idx,
    fps_text="",
):
    h, w = frame.shape[:2]

    regional = density_info.get("regional")
    if regional:
        cell_w = regional["cell_w"]
        cell_h = regional["cell_h"]
        counts = regional["counts"]
        for r in range(len(counts)):
            for c in range(len(counts[0])):
                x0 = int(c * cell_w)
                y0 = int(r * cell_h)
                x1 = int((c + 1) * cell_w)
                y1 = int((r + 1) * cell_h)
                count = counts[r][c]
                if count > 0:
                    is_hot = (r, c, count) in regional["hotspots"]
                    overlay_color = COLOR_HOTSPOT if is_hot else (60, 60, 60)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), overlay_color, -1)
                    alpha = 0.25 if is_hot else 0.08
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (90, 90, 90), 1)
                cv2.putText(
                    frame,
                    str(count),
                    (x0 + 5, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

    runners = 0
    for p in person_tracks:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        spd = speeds.get(p["id"], 0.0)
        is_running = spd >= running_speed_thresh
        if is_running:
            runners += 1
        color = COLOR_RUNNING if is_running else COLOR_WALK
        thickness = 3 if is_running else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"#{p['id']} {spd:.0f}px/s"
        cv2.putText(
            frame,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    total = density_info["total_people"]
    max_exp = density_info["max_expected"]
    ratio = density_info["ratio"]
    hud_lines = [
        f"Frame {frame_idx}  {fps_text}",
        f"People: {total} / {max_exp}  ({ratio*100:.0f}%)",
        f"Runners: {runners} / {total}",
    ]
    y = 25
    for line in hud_lines:
        _put_text(frame, line, (10, y))
        y += 25

    firing = [k for k, v in anomaly_status.items() if v == "firing"]
    if firing:
        msg = "!! ANOMALY: " + ", ".join(firing).upper() + " !!"
        cv2.rectangle(frame, (0, 0), (w, 50), COLOR_ALARM_BANNER, -1)
        _put_text(frame, msg, (10, 35), 0.8)
    elif density_info["global_state"] == "warning":
        cv2.rectangle(frame, (0, 0), (w, 40), COLOR_WARN_BANNER, -1)
        _put_text(
            frame,
            f"WARNING: high density ({total}/{max_exp})",
            (10, 28),
            0.7,
        )

    return frame


def draw_method2(frame, flow_result, anomaly_status, frame_idx, fps_text=""):
    h, w = frame.shape[:2]
    cell_mags = flow_result.get("cell_mags")

    if cell_mags is not None:
        rows, cols = cell_mags.shape
        cell_w = w / cols
        cell_h = h / rows

        max_mag = max(float(cell_mags.max()), 1.0)
        norm = cell_mags / max_mag

        overlay = frame.copy()
        for r in range(rows):
            for c in range(cols):
                v = norm[r, c]
                if v < 0.1:
                    continue
                x0 = int(c * cell_w)
                y0 = int(r * cell_h)
                x1 = int((c + 1) * cell_w)
                y1 = int((r + 1) * cell_h)
                red = int(255 * v)
                blue = int(255 * (1 - v))
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (blue, 0, red), -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    if flow_result.get("is_baseline_learning"):
        baseline_msg = "Baseline learning..."
    else:
        baseline_msg = f"Baseline mag: {flow_result['baseline_magnitude']:.2f}"

    hud = [
        f"Frame {frame_idx}  {fps_text}",
        f"Mean mag: {flow_result['mean_magnitude']:.2f}  ({baseline_msg})",
        f"High motion cells: {flow_result['high_motion_cells']} "
        f"({flow_result['high_motion_ratio']*100:.0f}%)",
        f"Direction entropy: {flow_result['mean_direction_entropy']:.2f}",
    ]
    y = 25
    for line in hud:
        _put_text(frame, line, (10, y))
        y += 25

    firing = [k for k, v in anomaly_status.items() if v == "firing"]
    if firing:
        cv2.rectangle(frame, (0, 0), (w, 50), COLOR_ALARM_BANNER, -1)
        _put_text(
            frame,
            "!! ANOMALY: " + ", ".join(firing).upper() + " !!",
            (10, 35),
            0.8,
        )

    return frame

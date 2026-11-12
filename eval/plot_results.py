"""
Produces academic-thesis-quality figures from benchmark results.

Outputs:
  - figures/summary_table.png         — P/R/F1/latency per method
  - figures/precision_recall_bar.png  — P/R/F1 bar
  - figures/latency_box.png           — latency distribution per method
  - figures/per_video_alarms.png      — videos × method alarm map
  - figures/realtime_fps.png          — FPS comparison per method
  - figures/inference_ms_box.png      — inference time per frame

Usage:
    python -m eval.plot_results \
        --json benchmark_results/anomaly_benchmark.json \
        --out-dir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_payload(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def fig_summary_table(summaries: list, out: Path):
    df = pd.DataFrame(summaries)
    cols = [
        "method",
        "TP",
        "FP",
        "FN",
        "TN",
        "Precision",
        "Recall",
        "F1",
        "AvgLatency_sec",
        "MedianLatency_sec",
    ]
    df = df[[c for c in cols if c in df.columns]]
    fig, ax = plt.subplots(figsize=(min(2 + 0.9 * len(df.columns), 14), 1.0 + 0.5 * len(df)))
    ax.axis("off")
    table = ax.table(
        cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    plt.title("Crowd Anomaly Detection — Method Comparison Summary", pad=12)
    plt.tight_layout()
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()


def fig_pr_bar(summaries: list, out: Path):
    methods = [s["method"] for s in summaries]
    P = [s["Precision"] for s in summaries]
    R = [s["Recall"] for s in summaries]
    F = [s["F1"] for s in summaries]
    x = np.arange(len(methods))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(methods) + 4), 4.5))
    ax.bar(x - w, P, w, label="Precision", color="#3b7dd8")
    ax.bar(x, R, w, label="Recall", color="#d8693b")
    ax.bar(x + w, F, w, label="F1", color="#3bd87d")
    for i, vals in enumerate(zip(P, R, F)):
        for j, v in enumerate(vals):
            ax.text(
                x[i] + (j - 1) * w,
                v + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 — Method comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def fig_latency_box(per_video: list, out: Path):
    df = pd.DataFrame(per_video)
    if df.empty or "latency_sec" not in df.columns:
        return
    methods = sorted(df["method"].unique())
    data = [df[df["method"] == m]["latency_sec"].dropna().tolist() for m in methods]
    if not any(data):
        return
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(methods) + 3), 4.5))
    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True)
    palette = ["#3b7dd8", "#d8693b", "#3bd87d", "#d83b9c"]
    for patch, color in zip(bp["boxes"], palette[: len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Detection latency (s)\n(alarm delay relative to GT start)")
    ax.set_title("Alarm trigger delay — distribution per method")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def fig_per_video_alarms(per_video: list, out: Path):
    df = pd.DataFrame(per_video)
    if df.empty:
        return
    pivot = df.pivot_table(
        index="video",
        columns="method",
        values="any_alarm",
        aggfunc="first",
    ).fillna(False)
    pivot = pivot.astype(int)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(pivot.columns) + 4), 0.4 * len(pivot) + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(
                j,
                i,
                "ALARM" if pivot.values[i, j] else "—",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if pivot.values[i, j] else "gray",
            )
    plt.colorbar(im, ax=ax, label="Did an alarm fire?")
    ax.set_title("Video × Method alarm map")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def fig_realtime_fps(per_video: list, out: Path):
    df = pd.DataFrame(per_video)
    if df.empty or "realtime_fps" not in df.columns:
        return
    pivot = df.pivot_table(
        index="video", columns="method", values="realtime_fps", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(pivot) + 3), 4.5))
    pivot.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_ylabel("Realtime FPS (total frames / wall time)")
    ax.set_title("Realtime performance per method")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def fig_inference_ms(per_video: list, out: Path):
    df = pd.DataFrame(per_video)
    if df.empty or "median_inference_ms" not in df.columns:
        return
    methods = sorted(df["method"].unique())
    data = [
        df[df["method"] == m]["median_inference_ms"].dropna().tolist()
        for m in methods
    ]
    if not any(data):
        return
    fig, ax = plt.subplots(figsize=(max(5, 1.3 * len(methods) + 3), 4.5))
    bp = ax.boxplot(data, tick_labels=methods, patch_artist=True)
    palette = ["#3b7dd8", "#d8693b", "#3bd87d", "#d83b9c"]
    for patch, color in zip(bp["boxes"], palette[: len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Median inference (ms / frame)")
    ax.set_title("Inference time per frame — distribution per method")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="path to anomaly_benchmark.json")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    payload = load_payload(Path(args.json))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries = payload.get("summaries", [])
    per_video = payload.get("per_video", [])

    if summaries:
        fig_summary_table(summaries, out / "summary_table.png")
        fig_pr_bar(summaries, out / "precision_recall_bar.png")
    if per_video:
        fig_latency_box(per_video, out / "latency_box.png")
        fig_per_video_alarms(per_video, out / "per_video_alarms.png")
        fig_realtime_fps(per_video, out / "realtime_fps.png")
        fig_inference_ms(per_video, out / "inference_ms_box.png")

    print(f"[OK] Figures ready: {out.resolve()}")


if __name__ == "__main__":
    main()

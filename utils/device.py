"""
Automatic selection of the compute device (CUDA / MPS / CPU).

Usage:
    from utils.device import select_device, describe_device
    device = select_device("auto")           # "auto" | "cpu" | "mps" | "cuda"
    print(describe_device(device))
"""
from __future__ import annotations

import sys
import warnings


def select_device(preference: str = "auto") -> str:
    """
    preference ∈ {"auto", "cpu", "mps", "cuda", "cuda:0", ...}
    auto: cuda:0 if CUDA is available, mps if Apple Silicon MPS is available, otherwise cpu.
    """
    pref = (preference or "auto").lower().strip()

    try:
        import torch  # noqa: F401
    except ImportError:
        warnings.warn("[device] torch is not installed → cpu selected.")
        return "cpu"
    import torch

    if pref == "cpu":
        return "cpu"

    if pref.startswith("cuda"):
        if torch.cuda.is_available():
            return pref if ":" in pref else "cuda:0"
        warnings.warn("[device] CUDA was requested but not found → cpu fallback.")
        return "cpu"

    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        warnings.warn("[device] MPS was requested but not found → cpu fallback.")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    try:
        import torch
    except ImportError:
        return f"{device} (torch missing)"

    if device.startswith("cuda"):
        idx = 0 if ":" not in device else int(device.split(":")[1])
        try:
            name = torch.cuda.get_device_name(idx)
            mem = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
            return f"{device} → {name} ({mem:.1f} GB)"
        except Exception:
            return f"{device} → CUDA device"
    if device == "mps":
        return f"mps → Apple Silicon GPU (Metal Performance Shaders), Python {sys.version.split()[0]}"
    return f"{device} (CPU)"

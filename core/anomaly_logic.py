"""
Alarm state machine.

Raw anomaly signals (they can flicker every frame) → stable alarm output.
  - persistence: the anomaly must last at least T seconds
  - cooldown: after an alarm fires, it does not fire again for T seconds

Alarm types:
  - "collective_panic"     (Method 1: running-person ratio is high)
  - "global_density"       (Method 1: total people exceeded the max)
  - "regional_density"     (Method 1: overcrowding in one cell)
  - "flow_panic"           (Method 2: high optical flow + coherent direction)
  - "flow_sudden_motion"   (Method 2: sudden motion jump)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict


class AnomalyStateMachine:
    def __init__(self, persistence_seconds: float, cooldown_seconds: float, fps: float):
        fps_eff = max(fps, 1.0)
        self.persistence_frames = max(1, int(persistence_seconds * fps_eff))
        self.cooldown_frames = max(1, int(cooldown_seconds * fps_eff))
        self.signal_streak: Dict[str, int] = defaultdict(int)
        self.last_alarm_frame: Dict[str, int] = defaultdict(lambda: -10 ** 9)
        self.active_alarms: set = set()

    def update(self, frame_idx: int, raw_signals: Dict[str, bool]) -> Dict[str, str]:
        """raw_signals: {alarm_type: bool}; Returns {alarm_type: 'firing' | 'idle'}."""
        statuses: Dict[str, str] = {}
        for alarm_type, is_signaling in raw_signals.items():
            if is_signaling:
                self.signal_streak[alarm_type] += 1
            else:
                self.signal_streak[alarm_type] = 0
                self.active_alarms.discard(alarm_type)

            cooldown_active = (
                frame_idx - self.last_alarm_frame[alarm_type]
            ) < self.cooldown_frames

            if (
                self.signal_streak[alarm_type] >= self.persistence_frames
                and not cooldown_active
            ):
                self.last_alarm_frame[alarm_type] = frame_idx
                self.active_alarms.add(alarm_type)

            statuses[alarm_type] = "firing" if alarm_type in self.active_alarms else "idle"

        return statuses

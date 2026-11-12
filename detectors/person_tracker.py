"""
Ultralytics YOLO + built-in ByteTrack wrapper for person detection + tracking.

Ultralytics' track() method handles YOLO + ByteTrack/BoT-SORT in a single call.
It returns a track_id for every detection. In a crowd this is faster and more
accurate than an external DeepSORT.

Device selection is automatic: if device="auto", the order is CUDA > MPS > CPU.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from ultralytics import YOLO


class PersonTracker:
    """YOLO + built-in tracker. Person class only."""

    def __init__(
        self,
        model_path: str,
        device: str,
        conf: float,
        iou: float,
        person_id: int,
        tracker: str,
        img_size: int = 640,
        max_det: int = 300,
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.person_id = person_id
        self.tracker_yaml = tracker
        self.img_size = int(img_size)
        self.max_det = int(max(1, min(1000, max_det)))

    def track(self, frame: np.ndarray) -> List[Dict]:
        """
        Returns list of dicts:
            {'id': int, 'bbox': [x1,y1,x2,y2], 'center': (cx,cy), 'conf': float}
        """
        sz = (max(320, min(1280, self.img_size)) // 32) * 32
        results = self.model.track(
            source=frame,
            classes=[self.person_id],
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            tracker=self.tracker_yaml,
            persist=True,
            verbose=False,
            imgsz=sz,
            max_det=self.max_det,
        )[0]

        out: List[Dict] = []
        if results.boxes is None or len(results.boxes) == 0:
            return out

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        ids = results.boxes.id
        if ids is None:
            # Tracks not yet confirmed — counting is meaningless for this frame.
            return out
        ids = ids.cpu().numpy().astype(int)

        for box, c, tid in zip(boxes, confs, ids):
            x1, y1, x2, y2 = box.tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            out.append(
                {
                    "id": int(tid),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": (float(cx), float(cy)),
                    "conf": float(c),
                }
            )
        return out

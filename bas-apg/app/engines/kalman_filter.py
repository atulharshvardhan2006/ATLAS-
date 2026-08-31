"""
BAS-APG — Kalman Filter for Bounding Boxes

Provides temporal smoothing and mathematical trajectory prediction
for YOLO bounding boxes to handle extreme occlusions and jitter.
"""

import cv2
import numpy as np


class BBoxKalmanFilter:
    """Kalman filter for a single bounding box tracking (x_center, y_center, w, h)."""

    def __init__(self):
        # 8 state variables: cx, cy, w, h, v_cx, v_cy, v_w, v_h
        # 4 measurement variables: cx, cy, w, h
        self.kf = cv2.KalmanFilter(8, 4)

        # State transition matrix (A)
        # x_k = x_{k-1} + v_{k-1} * dt (assuming dt=1)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],  # cx
                [0, 1, 0, 0, 0, 1, 0, 0],  # cy
                [0, 0, 1, 0, 0, 0, 1, 0],  # w
                [0, 0, 0, 1, 0, 0, 0, 1],  # h
                [0, 0, 0, 0, 1, 0, 0, 0],  # v_cx
                [0, 0, 0, 0, 0, 1, 0, 0],  # v_cy
                [0, 0, 0, 0, 0, 0, 1, 0],  # v_w
                [0, 0, 0, 0, 0, 0, 0, 1],  # v_h
            ],
            np.float32,
        )

        # Measurement matrix (H)
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            np.float32,
        )

        # Process noise covariance (Q)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2

        # Measurement noise covariance (R)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1

        # Error covariance (P)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * 1.0

        self.is_initialized = False

    def predict(self) -> list[float]:
        """Predict the next bounding box state."""
        pred = self.kf.predict()
        cx, cy, w, h = pred[0, 0], pred[1, 0], pred[2, 0], pred[3, 0]
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def update(self, bbox: list[float]):
        """Update the filter with a new measurement."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        measurement = np.array([[cx], [cy], [w], [h]], np.float32)

        if not self.is_initialized:
            self.kf.statePost = np.array(
                [[cx], [cy], [w], [h], [0], [0], [0], [0]], np.float32
            )
            self.is_initialized = True
        else:
            self.kf.correct(measurement)


class MultiObjectKalmanTracker:
    """Manages multiple Kalman filters for tracked objects."""

    def __init__(self):
        self.filters: dict[int, BBoxKalmanFilter] = {}

    def update(self, detections: list[dict]) -> list[dict]:
        """Update filters with detections and return smoothed detections."""
        smoothed_detections = []

        current_ids = set()
        for det in detections:
            track_id = det.get("track_id", -1)
            if track_id == -1:
                smoothed_detections.append(det)
                continue

            current_ids.add(track_id)
            if track_id not in self.filters:
                self.filters[track_id] = BBoxKalmanFilter()

            kf = self.filters[track_id]

            # Predict and correct
            kf.predict()
            kf.update(det["bbox"])

            # Get smoothed state
            smoothed_state = kf.kf.statePost
            cx, cy, w, h = (
                smoothed_state[0, 0],
                smoothed_state[1, 0],
                smoothed_state[2, 0],
                smoothed_state[3, 0],
            )
            smoothed_bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

            new_det = det.copy()
            new_det["bbox"] = smoothed_bbox
            smoothed_detections.append(new_det)

        # Cleanup lost tracks
        for tid in list(self.filters.keys()):
            if tid not in current_ids:
                del self.filters[tid]

        return smoothed_detections

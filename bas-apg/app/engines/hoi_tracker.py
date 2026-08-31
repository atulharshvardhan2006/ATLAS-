"""
BAS-APG — Hand-Object Interaction (HOI) Tracker

Determines hand-object interactions using Velocity Correlation (Pearson's r)
and Real-World Metric Space distances.

State machine per (hand, object) pair:
    FAR ──(distance < threshold)──→ NEAR ──(high velocity correlation)──→ HELD
     ↑                                ↓
     └──────(distance > threshold)────┘
"""

import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr

from app.core.config import get_settings


class HOITracker:
    """Tracks hand-object interactions across frames using Velocity Correlation.

    Args:
        contact_threshold_mm: Distance in millimeters below which hand is "NEAR".
        velocity_correlation_threshold: Pearson's r threshold (e.g., 0.7) for HELD.
        history_size: Number of frames to track for velocity calculation (default 15).
        calibration_path: Path to camera intrinsic calibration JSON.
    """

    def __init__(
        self,
        contact_threshold_mm: int = 100,
        velocity_correlation_threshold: float = 0.7,
        history_size: int = 15,
        calibration_path: str = "data/camera_calibration.json",
    ):
        self.contact_threshold = contact_threshold_mm
        self.corr_threshold = velocity_correlation_threshold
        self.history_size = history_size

        # Load camera calibration if available
        self.camera_matrix = None
        self.nominal_depth_mm = 500.0  # Assume objects are ~50cm away if no depth info
        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r") as f:
                    calib = json.load(f)
                    self.camera_matrix = np.array(calib["camera_matrix"])
                    print(
                        f"[HOI] Loaded camera calibration: fx={self.camera_matrix[0,0]:.1f}, fy={self.camera_matrix[1,1]:.1f}"
                    )
            except Exception as e:
                print(f"[HOI] WARNING: Failed to load calibration: {e}")

        # Track history for velocity correlation
        # hand_history: list of (x, y) in metric space
        self.hand_history = []
        # obj_history: {track_id: list of (x, y) in metric space}
        self.obj_history: dict[int, list[tuple[float, float]]] = defaultdict(list)

        # Track interaction state: {track_id: "FAR"|"NEAR"|"HELD"}
        self.interaction_states: dict[int, str] = defaultdict(lambda: "FAR")

        self.settings = get_settings()
        self.hand_velocity_history = []

        # Track class names for held objects: {track_id: class_name}
        self._class_map: dict[int, str] = {}

    def _pixel_to_metric(
        self, px: float, py: float, depth_mm: float
    ) -> tuple[float, float]:
        """Convert 2D pixel coordinate to 3D metric coordinate (X, Y) given Z depth."""
        if self.camera_matrix is not None:
            fx = self.camera_matrix[0, 0]
            fy = self.camera_matrix[1, 1]
            cx = self.camera_matrix[0, 2]
            cy = self.camera_matrix[1, 2]
            x_mm = (px - cx) * depth_mm / fx
            y_mm = (py - cy) * depth_mm / fy
            return (x_mm, y_mm)
        else:
            # Fallback: assume rough conversion (e.g. 1 pixel = 1 mm at 50cm)
            return (float(px), float(py))

    def _calculate_velocity(
        self, history: list[tuple[float, float, float]]
    ) -> list[float]:
        """Calculate velocity magnitudes from position history in 3D."""
        if len(history) < 2:
            return []
        velocities = []
        for i in range(1, len(history)):
            dx = history[i][0] - history[i - 1][0]
            dy = history[i][1] - history[i - 1][1]
            dz = history[i][2] - history[i - 1][2]
            velocities.append(math.sqrt(dx**2 + dy**2 + dz**2))
        return velocities

    def compute_interactions(
        self, palm_center: tuple[int, int] | None, detections: list[dict]
    ) -> list[dict]:
        """Compute interaction state for each detected object using Velocity Correlation.

        Args:
            palm_center: (x, y) pixel coordinates of the palm.
            detections: list of dicts with keys: class, track_id, bbox, confidence.
            depth_estimator: DepthEstimator instance.
            depth_map: Estimated depth map from the estimator.

        Returns:
            list of dicts with keys: class, track_id, state (FAR/NEAR/HELD), distance.
        """
        results = []
        current_track_ids = set()

        # Update hand history
        palm_z = self.nominal_depth_mm
        if palm_center:
            if depth_estimator and depth_map is not None:
                palm_z = depth_estimator.get_z_value(
                    depth_map, palm_center[0], palm_center[1]
                )
            palm_metric = self._pixel_to_metric(palm_center[0], palm_center[1], palm_z)
            self.hand_history.append((palm_metric[0], palm_metric[1], palm_z))
            if len(self.hand_history) > self.history_size:
                self.hand_history.pop(0)
        else:
            self.hand_history.clear()

        hand_velocities = self._calculate_velocity(self.hand_history)

        for det in detections:
            track_id = det.get("track_id", -1)
            if track_id == -1:
                continue

            current_track_ids.add(track_id)
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            class_name = det["class"]

            self._class_map[track_id] = class_name

            # Calculate object center and metric position
            obj_cx = int((bbox[0] + bbox[2]) / 2)
            obj_cy = int((bbox[1] + bbox[3]) / 2)

            obj_z = self.nominal_depth_mm
            obj_metric = self._pixel_to_metric(obj_cx, obj_cy, obj_z)

            self.obj_history[track_id].append((obj_metric[0], obj_metric[1], obj_z))
            if len(self.obj_history[track_id]) > self.history_size:
                self.obj_history[track_id].pop(0)

            if not palm_center or len(self.hand_history) < 5:
                # Not enough history or no hand
                self.interaction_states[track_id] = "FAR"
                results.append(
                    {
                        "class": class_name,
                        "track_id": track_id,
                        "state": "FAR",
                        "distance": float("inf"),
                    }
                )
                continue

            # Calculate True 3D Metric Distance
            distance_mm = math.sqrt(
                (palm_metric[0] - obj_metric[0]) ** 2
                + (palm_metric[1] - obj_metric[1]) ** 2
                + (palm_z - obj_z) ** 2
            )

            if distance_mm < self.contact_threshold:
                # Hand is NEAR. Now check Velocity Correlation
                obj_velocities = self._calculate_velocity(self.obj_history[track_id])

                # We need equal length velocity histories for correlation
                min_len = min(len(hand_velocities), len(obj_velocities))
                if min_len >= 5:
                    h_vel = hand_velocities[-min_len:]
                    o_vel = obj_velocities[-min_len:]

                    # Prevent division by zero in correlation if velocities are constant
                    if np.std(h_vel) > 1e-5 and np.std(o_vel) > 1e-5:
                        r, _ = pearsonr(h_vel, o_vel)
                    else:
                        # If both are barely moving, they might be held stationary
                        r = (
                            1.0
                            if np.mean(h_vel) < 5.0 and np.mean(o_vel) < 5.0
                            else 0.0
                        )

                    if r >= self.corr_threshold:
                        self.interaction_states[track_id] = "HELD"
                    else:
                        self.interaction_states[track_id] = "NEAR"
                else:
                    self.interaction_states[track_id] = "NEAR"
            else:
                self.interaction_states[track_id] = "FAR"

            results.append(
                {
                    "class": class_name,
                    "track_id": track_id,
                    "state": self.interaction_states[track_id],
                    "distance": round(distance_mm, 1),
                }
            )

        # Cleanup lost tracks from history
        for tid in list(self.obj_history.keys()):
            if tid not in current_track_ids:
                del self.obj_history[tid]
                if tid in self.interaction_states:
                    del self.interaction_states[tid]

        return results

    def get_held_objects(self) -> list[str]:
        """Return class names of all currently HELD objects."""
        return [
            self._class_map.get(tid, "unknown")
            for tid, state in self.interaction_states.items()
            if state == "HELD"
        ]

    def update_immobility_status(self, current_velocity_magnitude: float) -> str:
        """
        Immobility Guardian: Checks if hand variance drops below threshold for 30s.
        """
        import numpy as np

        self.hand_velocity_history.append(current_velocity_magnitude)

        # Keep buffer to exactly APG_IMMOBILITY_FRAMES
        if len(self.hand_velocity_history) > self.settings.immobility_frames:
            self.hand_velocity_history.pop(0)

            # Calculate mathematical variance
            velocity_variance = np.var(self.hand_velocity_history)

            if velocity_variance < self.settings.immobility_variance_threshold:
                return "CREW_EMERGENCY_IMMOBILITY"
        return "NOMINAL"

    def reset(self):
        """Reset all tracking state."""
        self.hand_history.clear()
        self.obj_history.clear()
        self.interaction_states.clear()
        self._class_map.clear()

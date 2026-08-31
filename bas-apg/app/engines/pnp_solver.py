"""
BAS-APG — Perspective-n-Point (PnP) Solver

Maps 2D image keypoints to assumed 3D object dimensions to
estimate pitch, yaw, and roll of rigid tools in microgravity.
"""

import json
import os

import cv2
import numpy as np


class PnPSolver:
    """Solves 6-DoF pose for known rigid tools."""

    def __init__(self, calibration_path: str = "data/camera_calibration.json"):
        self.camera_matrix = None
        self.dist_coeffs = None

        if os.path.exists(calibration_path):
            try:
                with open(calibration_path, "r") as f:
                    calib = json.load(f)
                    self.camera_matrix = np.array(calib["camera_matrix"])
                    self.dist_coeffs = np.array(calib["dist_coeffs"])
            except Exception as e:
                print(f"[PnP] WARNING: Failed to load calibration: {e}")

        if self.camera_matrix is None:
            # Fallback approximations for a 640x480 webcam
            self.camera_matrix = np.array(
                [[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype="double"
            )
            self.dist_coeffs = np.zeros((4, 1))

        # Assumed dimensions (in mm) for tools to create 3D object points
        # Assuming a standard box shape: 100x100x100mm
        self.object_points = np.array(
            [
                [-50, -50, 0],  # Top-left
                [50, -50, 0],  # Top-right
                [50, 50, 0],  # Bottom-right
                [-50, 50, 0],  # Bottom-left
            ],
            dtype="double",
        )

    def estimate_pose(self, bbox: list[float]) -> dict | None:
        """Estimate 3D pose from a 2D bounding box.

        Args:
            bbox: [x1, y1, x2, y2]
        Returns:
            dict with pitch, yaw, roll in degrees, or None if failed.
        """
        x1, y1, x2, y2 = bbox

        # We assume the bounding box corners roughly map to the object corners.
        # This is a simplification. True PnP needs precise keypoints.
        image_points = np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
            ],
            dtype="double",
        )

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        from app.engines.kinematic_exporter import extract_kinematics

        return extract_kinematics(rotation_vector, translation_vector)

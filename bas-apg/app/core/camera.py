"""
BAS-APG — Threaded Camera Buffer

Runs cv2.VideoCapture in a dedicated background thread to prevent
OpenCV's synchronous C++ blocking calls from freezing the FastAPI
asyncio event loop.

Thread safety: uses threading.Lock to protect the shared frame buffer.
The FastAPI event loop calls get_frame() which returns instantly.
"""

import threading
import time

import cv2
import numpy as np

from app.core.config import get_settings


class CameraBuffer(threading.Thread):
    """Non-blocking camera capture in a background thread."""

    def __init__(
        self,
        camera_index: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ):
        super().__init__(daemon=True)
        settings = get_settings()

        self.camera_index = (
            camera_index if camera_index is not None else settings.camera_index
        )
        self.width = width if width is not None else settings.camera_width
        self.height = height if height is not None else settings.camera_height

        self._frame = None
        self._is_blind = False
        self._lock = threading.Lock()
        self._running = False
        self.fps = 0.0

    def start(self):
        self._running = True
        super().start()

    def get_frame(self) -> tuple[np.ndarray | None, bool]:
        """Returns the most recent frame and sensor blind status instantly."""
        with self._lock:
            if self._frame is None:
                return None, False
            return self._frame.copy(), self._is_blind

    def run(self):
        """Main capture loop — runs in background thread."""
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"[CAMERA] ERROR: Cannot open camera index {self.camera_index}")
            return

        prev_time = time.time()
        print(
            f"[CAMERA] Started capture: index={self.camera_index}, {self.width}x{self.height}"
        )

        while self._running:
            ret, frame = cap.read()

            if ret:
                # Phase 20: Sensor Gating (Kalman Drift Trap Fix)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

                white_pixels = np.sum(hist[240:])
                total_pixels = frame.shape[0] * frame.shape[1]

                is_blind = False
                if total_pixels > 0 and (white_pixels / total_pixels) > 0.85:
                    is_blind = True

                with self._lock:
                    self._frame = frame
                    self._is_blind = is_blind

                    now = time.time()
                    dt = now - prev_time
                    if dt > 0:
                        self.fps = 1.0 / dt
                    prev_time = now
            else:
                time.sleep(0.01)

            time.sleep(0.001)

        cap.release()
        print("[CAMERA] Capture stopped and released.")

    def stop(self):
        self._running = False

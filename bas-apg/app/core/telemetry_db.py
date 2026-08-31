"""
BAS-APG — SQLite Telemetry Buffer

Thread-safe SQLite wrapper to silently buffer ML frames
before flushing them to a Protobuf binary payload.
"""

import json
import os
import sqlite3
import threading


class TelemetryDB:
    def __init__(self, db_path: str = "data/telemetry.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    latency_ms INTEGER,
                    fps INTEGER,
                    fsm_state TEXT,
                    detections TEXT,
                    interactions TEXT,
                    kinematics TEXT
                )
            """)
            conn.commit()
            conn.close()

    def insert_frame(self, frame_data: dict):
        """Buffer a single frame payload (stripping the image data)."""
        timestamp = frame_data.get("timestamp", 0.0)
        latency_ms = frame_data.get("latency_ms", 0)
        fps = frame_data.get("fps", 0)

        fsm_state = json.dumps(frame_data.get("state", {}))
        detections = json.dumps(frame_data.get("detections", []))
        interactions = json.dumps(frame_data.get("interactions", []))
        kinematics = json.dumps(frame_data.get("kinematics", {}))

        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO telemetry_frames 
                (timestamp, latency_ms, fps, fsm_state, detections, interactions, kinematics)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    latency_ms,
                    fps,
                    fsm_state,
                    detections,
                    interactions,
                    kinematics,
                ),
            )
            conn.commit()
            conn.close()

    def extract_and_clear(self) -> list[dict]:
        """Extract all buffered frames and clear the table."""
        frames = []
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT timestamp, latency_ms, fps, fsm_state, detections, interactions, kinematics FROM telemetry_frames ORDER BY id ASC"
            )
            rows = cursor.fetchall()

            for row in rows:
                frames.append(
                    {
                        "timestamp": row[0],
                        "latency_ms": row[1],
                        "fps": row[2],
                        "fsm_state": json.loads(row[3]) if row[3] else {},
                        "detections": json.loads(row[4]) if row[4] else [],
                        "interactions": json.loads(row[5]) if row[5] else [],
                        "kinematics": json.loads(row[6]) if row[6] else {},
                    }
                )

            cursor.execute("DELETE FROM telemetry_frames")
            conn.commit()
            conn.close()

        return frames

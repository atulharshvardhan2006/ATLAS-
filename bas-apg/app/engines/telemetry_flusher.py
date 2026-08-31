"""
BAS-APG — Asynchronous Telemetry Flusher

Runs in a background thread upon experiment completion.
Pulls logs from the SQLite DB and compiles them into a
<50KB Protobuf binary payload for IDSN transmission.
"""

import os
import threading

from app.core.telemetry_db import TelemetryDB


def generate_telemetry_burst(procedure_name: str, db: TelemetryDB):
    """
    Extracts telemetry from SQLite and serializes it to Protobuf.
    Designed to run in a background thread.
    """

    def _flush_task():
        print("\n[TELEMETRY] 🚀 Asynchronous Telemetry Flush Initiated...")
        try:
            from app.schemas import telemetry_pb2
        except ImportError:
            print(
                "[TELEMETRY] ERROR: Protobuf bindings missing. Run compile_proto.py first."
            )
            return

        frames = db.extract_and_clear()
        if not frames:
            print("[TELEMETRY] No telemetry frames found in buffer.")
            return

        proto_log = telemetry_pb2.TelemetryLog()
        proto_log.procedure_name = procedure_name

        for frame in frames:
            proto_frame = proto_log.frames.add()
            proto_frame.timestamp = frame.get("timestamp", 0.0)
            proto_frame.latency_ms = frame.get("latency_ms", 0)
            proto_frame.fps = frame.get("fps", 0)

            state = frame.get("fsm_state", {})
            proto_frame.fsm_state.current_step_id = state.get("current_step", "")
            proto_frame.fsm_state.status = state.get("status", "")
            proto_frame.fsm_state.start_time = str(state.get("step_start_time", ""))
            proto_frame.fsm_state.recovery_text = state.get("recovery_text", "")

            for det in frame.get("detections", []):
                proto_det = proto_frame.detections.add()
                proto_det.class_name = det.get("class", "")
                proto_det.confidence = det.get("confidence", 0.0)
                proto_det.track_id = det.get("track_id", 0)

                bbox = det.get("bbox", [0, 0, 0, 0])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    proto_det.bbox_cx = (x1 + x2) / 2
                    proto_det.bbox_cy = (y1 + y2) / 2
                    proto_det.bbox_w = x2 - x1
                    proto_det.bbox_h = y2 - y1

            for inter in frame.get("interactions", []):
                proto_int = proto_frame.interactions.add()
                proto_int.class_name = inter.get("class", "")
                proto_int.state = inter.get("state", "")
                proto_int.distance_mm = inter.get("distance", 0.0)

            # Add Kinematics
            kinematics = frame.get("kinematics")
            if kinematics:
                k_frame = proto_log.kinematic_stream.add()
                k_frame.timestamp = frame.get("timestamp", 0.0)

                for pt in kinematics.get("hand_joints", []):
                    j = k_frame.hand_joints.add()
                    j.x = pt.get("x", 0.0)
                    j.y = pt.get("y", 0.0)
                    j.z = pt.get("z", 0.0)

                for tool_name, t_data in kinematics.get("tools", {}).items():
                    j = k_frame.tools[tool_name]
                    j.x = t_data.get("x", 0.0)
                    j.y = t_data.get("y", 0.0)
                    j.z = t_data.get("z", 0.0)
                    j.pitch = t_data.get("pitch", 0.0)
                    j.yaw = t_data.get("yaw", 0.0)
                    j.roll = t_data.get("roll", 0.0)

        binary_payload = proto_log.SerializeToString()

        output_file = "data/telemetry_burst.bin"
        with open(output_file, "wb") as f:
            f.write(binary_payload)

        bin_size = os.path.getsize(output_file)

        print(f"[TELEMETRY] ✅ Burst Compiled: {output_file}")
        print(f"[TELEMETRY] Payload Size: {bin_size / 1024:.2f} KB (Ready for IDSN)")

    # Run in detached background thread
    t = threading.Thread(target=_flush_task, daemon=True)
    t.start()

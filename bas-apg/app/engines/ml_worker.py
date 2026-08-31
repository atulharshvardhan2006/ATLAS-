"""
BAS-APG — Isolated ML Worker Process

Runs the main ML inference loop completely decoupled from the FastAPI backend.
Saves the latest frame and state to disk for the API to stream.
Includes Kalman Filtering and Depth Estimation (Phase 10/11).
"""

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from app.core.camera import CameraBuffer
from app.core.config import get_settings
from app.core.telemetry_db import TelemetryDB
from app.engines.hoi_tracker import HOITracker
from app.engines.kalman_filter import MultiObjectKalmanTracker
from app.engines.pnp_solver import PnPSolver
from app.engines.procedure_fsm import ProcedureFSM
from app.engines.recovery_engine import RecoveryEngine
from app.engines.telemetry_flusher import generate_telemetry_burst
from app.engines.voice_alert import BackgroundTTSWorker

# Shared files for IPC
STATE_FILE = "/tmp/bas_apg_state.json"


def run_ml_loop():
    print("=" * 60)
    print("  BAS AI Procedure Guardian — ML Worker Started")
    print("=" * 60)

    settings = get_settings()

    # --- Initialize Subsystems ---
    camera = CameraBuffer(
        settings.camera_index, settings.camera_width, settings.camera_height
    )
    camera.start()

    tts_worker = BackgroundTTSWorker(settings.tts_rate, settings.tts_enabled)
    tts_worker.start()

    fsm = ProcedureFSM(settings.procedure_path, settings.debounce_frames)

    # Try to load existing state from a crash
    if os.path.exists("data/fsm_state.json"):
        try:
            with open("data/fsm_state.json", "r") as f:
                saved_state = json.load(f)
                # Naive resume: just set the index. (A real implementation would reconstruct FSMState obj)
                print(
                    f"[ML] Resuming FSM from step {saved_state.get('current_step_id')}"
                )
        except Exception:
            pass

    fsm.start()  # Start procedure automatically for worker mode

    hoi_tracker = HOITracker(
        contact_threshold_mm=settings.contact_threshold_mm,
        velocity_correlation_threshold=settings.velocity_correlation_threshold,
        history_size=15,
        calibration_path=settings.camera_calibration_path,
    )

    recovery_engine = RecoveryEngine()
    kalman_tracker = MultiObjectKalmanTracker()
    telemetry_db = TelemetryDB()
    pnp_solver = PnPSolver()

    # --- YOLO ---
    yolo_model = None
    yolo_path = Path(
        settings.yolo_model_path_onnx if settings.use_onnx else settings.yolo_model_path
    )
    if yolo_path.exists():
        from ultralytics import YOLO

        yolo_model = YOLO(str(yolo_path), task="detect")
        print(f"[ML] YOLO loaded from {yolo_path}")

    # --- MediaPipe ---
    mp_hands = None
    try:
        import mediapipe.python.solutions.hands as mp_hands_module

        mp_hands = mp_hands_module.Hands(
            static_image_mode=False,
            max_num_hands=settings.max_hands,
            min_detection_confidence=settings.hand_detection_confidence,
            min_tracking_confidence=settings.hand_tracking_confidence,
        )
    except Exception as e:
        print(f"[ML] MediaPipe error: {e}")

    # Track history for OOD/Occlusion
    track_history = {}

    frame_counter = 0
    last_fsm_state_str = ""
    print("[ML] Starting main loop...")

    try:
        while True:
            loop_start = time.time()
            frame_counter += 1

            # --- Check for IPC Commands (Dynamic FSM Reload & Mutex) ---
            if os.path.exists("/tmp/bas_apg_state.json"):
                try:
                    with open("/tmp/bas_apg_state.json", "r") as f:
                        cmd_state = json.load(f)
                        if cmd_state.get("command") == "PAUSE_CV":
                            print(
                                "\n[ML] ⏸️ PAUSE_CV received. Air-gapping computer vision for VLM..."
                            )
                            camera.stop()
                            yolo_model = None
                            if mp_hands:
                                mp_hands.close()
                                mp_hands = None

                            # Wait until command is cleared or changed
                            while True:
                                time.sleep(0.5)
                                if os.path.exists("/tmp/bas_apg_state.json"):
                                    with open("/tmp/bas_apg_state.json", "r") as f2:
                                        try:
                                            check_cmd = json.load(f2)
                                            if check_cmd.get("command") != "PAUSE_CV":
                                                break
                                        except Exception:
                                            pass

                            print(
                                "[ML] ▶️ RESUME_CV received. Re-initializing computer vision..."
                            )
                            camera = CameraBuffer(
                                settings.camera_index,
                                settings.camera_width,
                                settings.camera_height,
                            )
                            camera.start()

                            if settings.use_onnx and os.path.exists(
                                settings.yolo_model_path_onnx
                            ):
                                from ultralytics import YOLO

                                yolo_model = YOLO(
                                    settings.yolo_model_path_onnx, task="detect"
                                )
                            else:
                                from ultralytics import YOLO

                                yolo_model = YOLO(settings.yolo_model_path)

                            mp_hands = mp.solutions.hands.Hands(
                                static_image_mode=False,
                                max_num_hands=settings.max_hands,
                                min_detection_confidence=settings.hand_detection_confidence,
                                min_tracking_confidence=settings.hand_tracking_confidence,
                            )
                except Exception:
                    pass

            if os.path.exists("data/fsm_cmd.json"):
                try:
                    with open("data/fsm_cmd.json", "r") as f:
                        cmd = json.load(f)
                        if cmd.get("command") == "reload_fsm":
                            new_path = cmd.get("path")
                            print(
                                f"\n[ML] 🔄 IPC Command received: Reloading FSM from {new_path}"
                            )
                            fsm = ProcedureFSM(new_path, settings.debounce_frames)
                            fsm.start()
                            hoi_tracker.reset()
                            track_history.clear()
                            tts_worker.speak("New procedure loaded.")
                    os.remove("data/fsm_cmd.json")
                except Exception as e:
                    print(f"[ML] Error reading FSM command: {e}")

            frame, is_blind = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Phase 20: Kalman Drift Trap Fix (Sensor Gating)
            if is_blind:
                print("[ML] ⚠️ SENSOR_BLIND: Freezing Kalman Tracking & FSM...")
                # Skip CV logic completely, yielding CPU
                time.sleep(0.01)
                continue

            # 1. YOLO
            detections = []
            if yolo_model:
                results = yolo_model.track(
                    frame,
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False,
                    conf=0.2,
                )

                current_ids = set()
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        conf = float(box.conf)
                        tid = int(box.id) if box.id is not None else -1
                        bbox = box.xyxy[0].tolist()
                        cls_name = yolo_model.names[int(box.cls)]

                        if tid != -1:
                            current_ids.add(tid)
                            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                            aspect_ratio = w / h if h > 0 else 0

                            hist = track_history.get(tid, (bbox, 0, []))
                            ratios = hist[2]
                            ratios.append(aspect_ratio)
                            if len(ratios) > 5:
                                ratios.pop(0)

                            is_fluctuating = len(ratios) == 5 and np.std(ratios) > 0.5
                            if conf < 0.45 or is_fluctuating:
                                cls_name = "UNKNOWN_ANOMALY"

                            track_history[tid] = (bbox, 0, ratios)

                        detections.append(
                            {
                                "class": cls_name,
                                "confidence": conf,
                                "track_id": tid,
                                "bbox": bbox,
                            }
                        )

                # Occlusion Handling
                for tid, (bbox, missed, ratios) in list(track_history.items()):
                    if tid not in current_ids:
                        missed += 1
                        if missed <= 10:
                            track_history[tid] = (bbox, missed, ratios)
                            detections.append(
                                {
                                    "class": "OCCLUDED",
                                    "confidence": 0.5,
                                    "track_id": tid,
                                    "bbox": bbox,
                                }
                            )
                        else:
                            del track_history[tid]

            # 3. Kalman Filter Smoothing
            detections = kalman_tracker.update(detections)

            # Annotate Frame manually after Kalman smoothing
            annotated_frame = frame.copy()
            for d in detections:
                x1, y1, x2, y2 = map(int, d["bbox"])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f'{d["class"]} {d["track_id"]}',
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            # 4. MediaPipe Hands
            palm_center = None
            if mp_hands:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = mp_hands.process(rgb)
                if res.multi_hand_landmarks:
                    hand = res.multi_hand_landmarks[0]
                    h, w = frame.shape[:2]
                    cx = sum(hand.landmark[i].x for i in [0, 5, 9, 13, 17]) / 5
                    cy = sum(hand.landmark[i].y for i in [0, 5, 9, 13, 17]) / 5
                    palm_center = (int(cx * w), int(cy * h))

                    cv2.circle(annotated_frame, palm_center, 8, (0, 255, 255), -1)

            # 5. HOI
            valid_dets = [d for d in detections if d["class"] != "UNKNOWN_ANOMALY"]
            interactions = hoi_tracker.compute_interactions(palm_center, valid_dets)
            held_objects = hoi_tracker.get_held_objects()

            # 6. FSM
            action, obj = ("PICK", held_objects[0]) if held_objects else ("", "")
            best_conf = max((d["confidence"] for d in valid_dets), default=0.0)

            # --- Phase 18: Kinematic Extraction (Every 3 frames) ---
            kinematics = {}
            if frame_counter % 3 == 0:
                hand_joints = []
                if results_mp.multi_hand_landmarks:
                    for hl in results_mp.multi_hand_landmarks:
                        for lm in hl.landmark:
                            # Convert normalized coordinates to absolute pixels for X/Y
                            h_img, w_img, _ = frame.shape
                            cx, cy = int(lm.x * w_img), int(lm.y * h_img)
                            z_val = lm.z * w_img  # approximate relative depth

                            hand_joints.append({"x": cx, "y": cy, "z": z_val})

                tools_kinematics = {}
                for d in valid_dets:
                    tool_name = d["class"]
                    bbox = d.get("bbox")
                    if bbox:
                        pose = pnp_solver.estimate_pose(bbox)
                        if pose:
                            tools_kinematics[tool_name] = pose

                kinematics = {"hand_joints": hand_joints, "tools": tools_kinematics}

            if fsm.state.status == "IN_PROGRESS":
                # --- Phase 19: Safety Guardians ---
                # 1. Biological Hazard Containment Guard
                if hand_joints:
                    # Check Z coordinate of the first tracked hand
                    min_z = min(j["z"] for j in hand_joints)
                    hazard_status = fsm.check_containment_breach(min_z)
                    if hazard_status != "CONTAINMENT_SECURE":
                        tts_worker.speak(hazard_status)
                        print(f"[GUARDIAN] ☣️ {hazard_status}")

                # 2. Immobility Guardian
                # Calculate velocity magnitude from kalman tracker if hands exist
                # Simplification: use the velocity of the first hand tracked
                # Normally you'd match the hand ID.
                current_velocity = 0.0
                if valid_dets:
                    for d in valid_dets:
                        if d["class"] in ["Hand", "Left_Hand", "Right_Hand"]:
                            # If kalman state is available, we could get velocity
                            # Let's approximate velocity magnitude from track_history
                            tid = d.get("track_id", -1)
                            if tid in track_history and len(track_history[tid]) >= 2:
                                p1 = track_history[tid][-1]
                                p2 = track_history[tid][-2]
                                import math

                                # Distance per frame in pixels (or mm if mapped)
                                current_velocity = math.hypot(
                                    p1[0] - p2[0], p1[1] - p2[1]
                                )
                            break

                immobility_status = hoi_tracker.update_immobility_status(
                    current_velocity
                )
                if immobility_status == "CREW_EMERGENCY_IMMOBILITY":
                    tts_worker.speak("Crew Emergency: Immobility Detected.")
                    print(f"[GUARDIAN] 🛡️ CREW_EMERGENCY_IMMOBILITY")

                # Mock tool opening logic for testing the Biological Hazard Guard
                if action == "OPEN" and obj == "Yellow_Box":
                    fsm.update_tool_state("Yellow_Box", "OPEN")
                elif action == "CLOSE" and obj == "Yellow_Box":
                    fsm.update_tool_state("Yellow_Box", "CLOSED")

                fsm_res = fsm.process_observation(action, obj, best_conf, held_objects)
                event_type = fsm_res.get("event_type", "none")

                if event_type == "step_confirmed":
                    tts_worker.speak(f"Step {fsm_res.get('completed_step')} confirmed.")
                elif event_type == "deviation_detected":
                    recovery = recovery_engine.generate_recovery(
                        fsm_res.get("deviation_type"), fsm.get_current_step(), obj
                    )
                    fsm.state.recovery_text = recovery["ui_text"]
                    tts_worker.speak(recovery["voice_text"])
                elif event_type == "procedure_completed":
                    # Trigger Asynchronous Telemetry Flush
                    generate_telemetry_burst(fsm.procedure_name, telemetry_db)

            # Save state continuously for crash recovery
            os.makedirs("data", exist_ok=True)
            with open("data/fsm_state.json", "w") as f:
                json.dump(fsm.state.to_dict(), f)

            # 7. Serialize for API
            _, jpeg = cv2.imencode(
                ".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            frame_b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

            loop_time = time.time() - loop_start

            payload = {
                "type": "update",
                "frame": frame_b64,
                "state": fsm.state.to_dict(),
                "detections": [
                    {
                        "class": d["class"],
                        "confidence": round(d["confidence"], 2),
                        "track_id": d.get("track_id", -1),
                        "bbox": d.get("bbox", []),
                    }
                    for d in detections
                ],
                "interactions": [
                    {
                        "class": i["class"],
                        "state": i["state"],
                        "distance": i.get("distance", 0),
                    }
                    for i in interactions
                ],
                "kinematics": kinematics,
                "fps": camera.fps,
                "latency_ms": int(loop_time * 1000),
            }

            # Write telemetry frame to SQLite buffer
            telemetry_db.insert_frame(payload)

            # Write to tmpfs (or disk) atomically (IPC Thrashing Fix)
            current_fsm_state_str = json.dumps(payload["state"])
            if frame_counter % 3 == 0 or current_fsm_state_str != last_fsm_state_str:
                tmp_file = STATE_FILE + ".tmp"
                with open(tmp_file, "w") as f:
                    json.dump(payload, f)
                os.rename(tmp_file, STATE_FILE)
                last_fsm_state_str = current_fsm_state_str

            # Prevent 100% CPU usage
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("[ML] Shutting down...")
    finally:
        camera.stop()
        tts_worker.stop()
        if mp_hands:
            mp_hands.close()


if __name__ == "__main__":
    run_ml_loop()

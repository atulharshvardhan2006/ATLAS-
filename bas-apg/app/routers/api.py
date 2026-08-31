"""
BAS-APG — API Routes (REST + WebSocket)

Backend-only endpoints (no frontend, no database):
  GET  /health              — Health check
  GET  /api/state           — Current FSM state
  POST /api/start           — Start/restart the procedure
  POST /api/acknowledge     — Acknowledge a deviation
  WS   /ws/live             — Live inference stream (JSON)

The /ws/live endpoint is the MAIN ML LOOP:
  Camera → YOLO → MediaPipe → HOI → FSM → Voice → WebSocket JSON
"""

import asyncio
import base64
import time

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

router = APIRouter()

# =============================================================================
# Module-level references (set by main.py during startup)
# =============================================================================
camera = None  # CameraBuffer instance
ws_manager = None  # WSManager instance
fsm = None  # ProcedureFSM instance
hoi_tracker = None  # HOITracker instance
recovery_engine = None  # RecoveryEngine instance
tts_worker = None  # BackgroundTTSWorker instance
yolo_model = None  # YOLO model instance
mp_hands = None  # MediaPipe Hands instance


# =============================================================================
# ML Helper Functions
# =============================================================================


def _get_palm_center(frame: np.ndarray) -> tuple[int, int] | None:
    """Extract palm center from frame using MediaPipe Hands.

    Returns (x, y) pixel coordinates or None if no hand detected.
    """
    if mp_hands is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        h, w = frame.shape[:2]

        # Palm center from MCP landmarks (more stable than fingertips)
        palm_indices = [0, 5, 9, 13, 17]
        cx = sum(hand.landmark[i].x for i in palm_indices) / len(palm_indices)
        cy = sum(hand.landmark[i].y for i in palm_indices) / len(palm_indices)

        return (int(cx * w), int(cy * h))

    return None

    if yolo_model is None:
        return [], frame.copy()

    # Track history for occlusion handling and aspect ratio fluctuation check
    if not hasattr(_detect_objects, "track_history"):
        _detect_objects.track_history = (
            {}
        )  # id -> (bbox, frames_missed, last_aspect_ratios)

    # Use a lower confidence threshold to catch potential OOD objects
    results = yolo_model.track(
        frame,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False,
        conf=0.2,
    )

    detections = []
    current_ids = set()

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf)
            track_id = int(box.id) if box.id is not None else -1
            bbox = box.xyxy[0].tolist()
            cls_name = yolo_model.names[int(box.cls)]

            if track_id != -1:
                current_ids.add(track_id)
                # Aspect ratio check
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                aspect_ratio = w / h if h > 0 else 0

                hist = _detect_objects.track_history.get(track_id, (bbox, 0, []))
                ratios = hist[2]
                ratios.append(aspect_ratio)
                if len(ratios) > 5:
                    ratios.pop(0)

                # Check for fluctuation
                is_fluctuating = False
                if len(ratios) == 5:
                    import numpy as np

                    if np.std(ratios) > 0.5:  # arbitrary threshold for wild fluctuation
                        is_fluctuating = True

                # OOD Rejection
                if conf < 0.45 or is_fluctuating:
                    cls_name = "UNKNOWN_ANOMALY"

                _detect_objects.track_history[track_id] = (bbox, 0, ratios)

            detections.append(
                {
                    "class": cls_name,
                    "confidence": conf,
                    "track_id": track_id,
                    "bbox": bbox,
                }
            )

    # Occlusion Handling: Inject missing tracks
    for tid, (bbox, missed, ratios) in list(_detect_objects.track_history.items()):
        if tid not in current_ids:
            missed += 1
            if missed <= 10:  # 10 frame grace period
                _detect_objects.track_history[tid] = (bbox, missed, ratios)
                # Re-inject the last known bounding box
                detections.append(
                    {
                        "class": (
                            "UNKNOWN" if missed > 5 else "OCCLUDED"
                        ),  # Can be specific if we stored class
                        "confidence": 0.5,
                        "track_id": tid,
                        "bbox": bbox,
                    }
                )
            else:
                del _detect_objects.track_history[tid]

    annotated = results[0].plot() if results else frame.copy()
    return detections, annotated


def _draw_hand_overlay(frame: np.ndarray, palm_center: tuple[int, int] | None):
    """Draw hand tracking overlay on frame."""
    if palm_center is None:
        return
    cv2.circle(frame, palm_center, 8, (0, 255, 255), -1)
    cv2.circle(frame, palm_center, 12, (0, 255, 255), 2)
    cv2.putText(
        frame,
        "PALM",
        (palm_center[0] + 15, palm_center[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )


def _infer_action(interactions: list[dict], held_objects: list[str]) -> tuple[str, str]:
    """Infer current action from HOI states.

    Simple heuristic: HELD → PICK (the held object).
    """
    if not held_objects:
        return "", ""
    return "PICK", held_objects[0]


# =============================================================================
# REST Endpoints
# =============================================================================


@router.get("/health")
async def health_check():
    """Health check — shows status of the decoupled backend."""
    import os

    state_exists = os.path.exists("/tmp/bas_apg_state.json")
    return {
        "status": "ok",
        "service": "bas-apg-api",
        "subsystems": {
            "ml_worker_active": state_exists,
        },
    }


@router.get("/api/state")
async def get_state():
    """Get current FSM state from ML Worker."""
    import json
    import os

    if not os.path.exists("/tmp/bas_apg_state.json"):
        return JSONResponse(
            {"error": "ML worker not initialized or state missing"}, status_code=503
        )

    try:
        with open("/tmp/bas_apg_state.json", "r") as f:
            data = json.load(f)
            return data.get("state", {})
    except Exception as e:
        return JSONResponse({"error": f"Failed to read state: {e}"}, status_code=500)


@router.post("/api/fsm/load")
async def load_fsm(request: Request):
    """Dynamically loads a new FSM generated by the VLM."""
    import json
    import os

    try:
        new_fsm = await request.json()

        # 1. Mutex Lock: Command ML Worker to pause CV loop
        with open("/tmp/bas_apg_state.json", "w") as f:
            json.dump({"command": "PAUSE_CV"}, f)

        import time

        # Simulate VLM taking time to load weights and parse PDF
        time.sleep(2.0)

        # Save the new FSM to disk for the worker to load
        os.makedirs("data", exist_ok=True)
        with open("data/active_procedure.json", "w") as f:
            json.dump(new_fsm, f, indent=4)

        # Write the IPC command for the ML worker to reload and resume
        with open("data/fsm_cmd.json", "w") as f:
            json.dump(
                {"command": "reload_fsm", "path": "data/active_procedure.json"}, f
            )

        return {
            "status": "success",
            "message": "FSM saved and reload command dispatched to ML Worker.",
        }
    except Exception as e:
        return JSONResponse({"error": f"Failed to load FSM: {e}"}, status_code=400)


@router.post("/api/start")
async def start_procedure():
    """Start or restart the experiment procedure (NOT SUPPORTED IN DECOUPLED MODE YET)."""
    return JSONResponse({"error": "Use Watchdog to restart worker"}, status_code=501)


@router.post("/api/acknowledge")
async def acknowledge_deviation():
    """Acknowledge a deviation (NOT SUPPORTED IN DECOUPLED MODE YET)."""
    return JSONResponse(
        {"error": "Action dispatching not implemented in decoupled IPC"},
        status_code=501,
    )


# =============================================================================
# WebSocket — Live ML Inference Stream
# =============================================================================


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Streams the latest ML payload generated by the isolated ML process."""
    await ws_manager.connect(websocket)

    import json
    import os

    STATE_FILE = "/tmp/bas_apg_state.json"
    last_mtime = 0

    try:
        while True:
            if os.path.exists(STATE_FILE):
                mtime = os.path.getmtime(STATE_FILE)
                if mtime > last_mtime:
                    try:
                        with open(STATE_FILE, "r") as f:
                            payload = json.load(f)
                            await websocket.send_json(payload)
                            last_mtime = mtime
                    except json.JSONDecodeError:
                        pass  # File might be mid-write, ignore and retry next loop

            await asyncio.sleep(0.03)  # roughly 30fps poll rate

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"[WS] Error: {e}")
        ws_manager.disconnect(websocket)

# Goal Description

Develop the AI Procedure Guardian (BAS-APG), a 100% offline, edge-computing cyber-physical system for the Bharatiya Antariksh Station (BAS). The system will monitor scientific payload experiments in microgravity using a threaded FastAPI backend, YOLOv8 for object detection, MediaPipe for 3D Hand Mesh Recovery (HMR), a deterministic Finite State Machine (FSM) for procedure tracking, and offline pyttsx3 text-to-speech for alerts.

## User Review Required

> [!IMPORTANT]
> **Edge Hardware Constraints:** The plan below targets the M4 MacBook Air (16GB) as the primary development/demo hub. If deploying directly to the Intel i3 (8GB), you MUST execute Phase 7 (ONNX quantization) before live demonstrations to avoid thermal throttling and out-of-memory errors.

> [!WARNING]
> **MediaPipe API Selection:** Based on research, we will use the Legacy Solutions API (`mediapipe.solutions.hands`) instead of the newer Tasks API. The legacy API bundles models directly inside the pip wheel, guaranteeing zero-network compliance out of the box without requiring external `.task` file downloads. 

## Open Questions

None currently. The PRD and research have provided a complete architectural blueprint.

## Proposed Changes

The following is a comprehensive, step-by-step roadmap for your team to build this project from zero to submission.

---

### Phase 1: Infrastructure & Boilerplate Setup
**Goal:** Initialize the environment, lock dependencies, and create the directory structure.

1. **Initialize Environment:**
   Create a Python virtual environment (e.g., using `conda` or `uv`) and install the exact researched packages to guarantee offline capability.
   ```bash
   conda create -n bas_ai_env python=3.10 -y
   conda activate bas_ai_env
   pip install fastapi>=0.115.0 uvicorn[standard]>=0.32.0 websockets>=13.0
   pip install ultralytics>=8.3.0 mediapipe>=0.10.0 opencv-python numpy
   pip install sqlalchemy pyttsx3 pydantic pydantic-settings aiofiles albumentations
   ```
2. **Scaffold Directory Structure:**
   Create the following structure in your repository:
   ```text
   bas-apg/
   ├── app/
   │   ├── main.py                 # FastAPI entry point
   │   ├── core/                   # Threaded Camera & WebSocket managers
   │   ├── models/                 # SQLAlchemy schemas (EventLog, ExperimentRun)
   │   ├── engines/                # FSM, TTS Voice, HOI Tracker
   │   ├── routers/                # FastAPI endpoints
   │   └── static/                 # HTML/JS/CSS for Cockpit UI
   ├── data/
   │   ├── procedures/             # JSON experiment schemas
   │   ├── models/                 # YOLOv8 .pt and .onnx weights
   │   └── evidence_logs/          # Output SQLite DB and keyframe images
   └── scripts/                    # Training and export utilities
   ```

---

### Phase 2: Data, Schemas & Synthetic Augmentation
**Goal:** Define the procedure rules and generate the microgravity-simulated dataset.

1. **Define Procedure Schema:**
   Create `data/procedures/red_yellow_box_experiment.json`.
   Define strict step IDs, expected tools (e.g., `main_box`, `red_box`), timeout thresholds, and valid next states.
2. **Physical Data Capture:**
   Record 50-80 video clips locally using your webcam. Perform correct executions, intentional skips, wrong object usage, and incomplete actions. 
3. **Data Augmentation (Simulating Microgravity):**
   Create a script using `albumentations` (`A.SafeRotate(limit=180, p=0.8)`) to rotate your dataset 360 degrees. This teaches YOLOv8 that objects in zero-G lack a fixed "up" vector.
4. **Annotation & Training:**
   Annotate bounding boxes using CVAT. Export in YOLO format and train a custom Nano model (`yolov8n.pt`). Place the resulting `best.pt` in `data/models/`.

---

### Phase 3: Perception Layer (The "Eyes")
**Goal:** Ingest video asynchronously and extract objects/hands without blocking.

1. **Threaded Camera Buffer (`app/core/camera.py`):**
   Implement `cv2.VideoCapture` inside a dedicated `threading.Thread`. This prevents OpenCV's synchronous C++ blocking calls from freezing the FastAPI `asyncio` event loop.
2. **Object Detection (`ultralytics`):**
   Load `yolov8n.pt` and run it on the latest frame from the buffer. Implement ByteTrack (`tracker="bytetrack.yaml"`) to assign persistent IDs to boxes.
3. **Hand Mesh Recovery (`mediapipe`):**
   Initialize `mp.solutions.hands.Hands`. Extract the 21 landmarks. Calculate the palm center using the average `(x,y)` of the landmarks, or specifically track fingertip indices (4, 8, 12, 16, 20).
4. **Hand-Object Interaction (HOI) Math:**
   Implement a function to calculate the Euclidean distance between the hand center and object bounding box centroids. If `distance < threshold` for > 5 frames, classify the state as `HELD`.

---

### Phase 4: Intelligence & Procedure FSM (The "Brain")
**Goal:** Enforce the strict sequence of events and handle deviations.

1. **FSM Initialization (`app/engines/procedure_fsm.py`):**
   Load the JSON schema. Track the `current_step_idx`.
2. **Temporal Debouncing:**
   Ensure an action (e.g., picking up the red box) is detected consistently for ~15 frames before transitioning the state to prevent noisy flickering.
3. **Deviation Logic:**
   - *Skip Check:* If detected action matches step `N+1` while step `N` is pending -> trigger `DEVIATION_SKIPPED`.
   - *Wrong Tool Check:* If action matches step `N` but object is wrong -> trigger `DEVIATION_WRONG_TOOL`.
4. **Recovery Engine:**
   When a deviation occurs, calculate the last valid state. Generate dynamic text (e.g., "Warning: Red box used instead of yellow box. Revert tool.") to pass to the Voice and UI engines.

---

### Phase 5: Cockpit & Network (FastAPI + WebSockets)
**Goal:** Serve the zero-latency Dashboard UI and stream processed video.

1. **WebSocket Manager (`app/core/ws_manager.py`):**
   Implement a connection manager to handle multiple clients. Broadcast using `asyncio`. 
2. **Video Streaming Route:**
   Create an endpoint `ws://localhost:8000/ws/live`. 
   Convert annotated `cv2` frames to JPEG, encode to Base64, and send as JSON (`{"type": "frame", "data": "base64...", "state": "S01"}`).
   *(Optimization for Edge: send raw binary JPEG bytes instead of base64 to save ~30% bandwidth).*
3. **Dashboard UI (`app/static/index.html`):**
   Build a sterile, high-contrast dark mode UI. Use JavaScript to connect to the WebSocket, paint the image to an `<img>` or `<canvas>` tag, and update the checklist and deviation alert box dynamically based on the JSON payload.

---

### Phase 6: Voice, Telemetry & Logging
**Goal:** Create offline audit trails and audio alerts.

1. **Threaded Voice Engine (`app/engines/voice_alert.py`):**
   > [!CAUTION]
   > `pyttsx3` is not thread-safe and will crash the event loop if run synchronously. 
   Implement a `BackgroundTTSWorker` class using `threading.Thread` and `queue.Queue`. Initialize `pyttsx3.init()` *inside* the worker thread's run loop.
2. **SQLite Evidence Engine (`app/models/database.py`):**
   Configure SQLAlchemy with `sqlite:///./data/evidence_logs/bas_apg.db`. Use `connect_args={"check_same_thread": False}`.
   Log every state transition (timestamp, step_idx, event_type, confidence).
3. **Evidence Snapshots:**
   When a step completes, save the raw `cv2` frame to disk (`S01_completed.jpg`) and log the path in the SQLite DB.
4. **Telemetry Burst Packager:**
   Write a script that queries the SQLite DB at the end of the run and dumps it into a highly compressed `.jsonl` file (<50KB target) simulating IDSN deep-space transmission constraints.

---

### Phase 7: Optimization & Demo Rehearsal
**Goal:** Prep for edge hardware and the SIH jury presentation.

1. **ONNX Export:**
   Write a script to export your trained YOLO model: `model.export(format="onnx", imgsz=640, simplify=True)`. Switch the backend to load `best.onnx` via `onnxruntime` using the `CPUExecutionProvider` for the Intel i3 test.
2. **Stress Testing:**
   Run the system continuously for 15 minutes. Monitor RAM usage (ensure it stays below 3.5GB on Mac, <2.0GB on i3) and verify no memory leaks occur in the camera buffer.
3. **Demo Choreography:**
   - **0:00 - 0:45:** Pitch the Microgravity / Comm Latency problem.
   - **0:45 - 1:30:** Show nominal execution (green checks, audio confirming steps).
   - **1:30 - 2:15:** *Crucial Step:* Intentionally grab the wrong box to trigger the deviation. Show the FSM catching it, the UI flashing red, and the TTS voice providing the recovery path.
   - **2:15 - 3:00:** Disconnect Wi-Fi to prove offline capability. Show the generated SQLite DB, keyframe images, and the compressed `.jsonl` telemetry payload.

## Verification Plan

### Automated Verification
None required initially; verification will be driven by visual confirmation of the FSM state changes.

### Manual Verification
The team will manually perform the physical analog experiments in front of the camera to verify:
1. Object detection works at varied angles (simulating microgravity).
2. The threaded camera buffer does not drop frames while pyttsx3 is speaking.
3. The WebSocket stream maintains < 100ms latency to the browser UI.

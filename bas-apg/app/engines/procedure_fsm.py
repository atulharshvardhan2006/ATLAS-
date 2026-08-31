"""
BAS-APG — Procedure Finite State Machine

Core engine that enforces the strict sequence of experiment steps.
Loads procedure definitions from JSON, tracks current state,
processes observations with temporal debouncing, and detects deviations.

CRITICAL RULES:
  - This module does NOT import any AI/CV libraries
  - It receives ONLY (action, object, confidence) tuples
  - Given the same observation sequence, it always produces the same result
  - Procedure definitions are loaded from JSON config, NEVER hard-coded

State diagram:
    IDLE → S01 → S02 → S03 → ... → S_N → COMPLETED
                                   ↗
                            DEVIATION (recoverable)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StepDefinition:
    """A single procedure step loaded from JSON schema."""

    step_id: str
    action: str
    object: str
    description: str
    timeout_seconds: float
    next_step: str | None
    confidence_threshold: float
    recovery_options: list[str]


@dataclass
class FSMState:
    """Current state of the FSM (serializable to JSON for the frontend)."""

    status: str = "IDLE"  # IDLE | IN_PROGRESS | COMPLETED | DEVIATION
    current_step_id: str | None = None
    current_step_index: int = 0
    expected_action: str = ""
    expected_object: str = ""
    confidence: float = 0.0
    step_start_time: float = 0.0
    completed_steps: list[str] = field(default_factory=list)
    deviation_type: str | None = None
    deviation_details: str = ""
    recovery_text: str = ""
    next_instruction: str = ""

    def to_dict(self) -> dict:
        """Serialize state for WebSocket transmission."""
        return {
            "status": self.status,
            "current_step": self.current_step_id,
            "current_step_index": self.current_step_index,
            "expected_action": self.expected_action,
            "expected_object": self.expected_object,
            "confidence": round(self.confidence, 3),
            "completed_steps": self.completed_steps.copy(),
            "total_steps": 0,  # Set by FSM externally
            "deviation_type": self.deviation_type,
            "deviation_details": self.deviation_details,
            "recovery_text": self.recovery_text,
            "next_instruction": self.next_instruction,
        }


# =============================================================================
# Procedure FSM
# =============================================================================


class ProcedureFSM:
    """Deterministic Finite State Machine for procedure tracking.

    Args:
        procedure_path: Path to the JSON procedure schema file.
        debounce_frames: Number of consecutive matching frames before confirming a step.
    """

    def __init__(self, procedure_path: str, debounce_frames: int = 15):
        self.steps: list[StepDefinition] = []
        self.state = FSMState()
        self.debounce_frames = debounce_frames
        self.procedure_id: str = ""
        self.procedure_name: str = ""
        self.deviation_count: int = 0

        # Debouncing state
        self._debounce_action: str = ""
        self._debounce_object: str = ""
        self._debounce_counter: int = 0

        # Phase 19: Biological Hazard Containment Guard
        from app.core.config import get_settings

        self.settings = get_settings()
        self.hazardous_containers = {"Yellow_Box": "CLOSED"}

        # Load procedure definition
        self._load_procedure(procedure_path)

    def _load_procedure(self, path: str):
        """Load procedure definition from JSON file."""
        p = Path(path)
        if not p.exists():
            print(f"[FSM] WARNING: Procedure file not found: {path}")
            print("[FSM] FSM will operate in passthrough mode.")
            return

        with open(p) as f:
            data = json.load(f)

        self.procedure_id = data["id"]
        self.procedure_name = data["name"]

        for step_data in data["steps"]:
            self.steps.append(
                StepDefinition(
                    step_id=step_data["step_id"],
                    action=step_data["action"],
                    object=step_data["object"],
                    description=step_data["description"],
                    timeout_seconds=step_data.get("timeout_seconds", 30),
                    next_step=step_data.get("next_step"),
                    confidence_threshold=step_data.get("confidence_threshold", 0.85),
                    recovery_options=step_data.get(
                        "recovery_options", ["voice_prompt"]
                    ),
                )
            )

        print(
            f"[FSM] Loaded procedure: {self.procedure_name} "
            f"({len(self.steps)} steps)"
        )

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def is_loaded(self) -> bool:
        return len(self.steps) > 0

    def start(self):
        """Begin the procedure (transition from IDLE to first step)."""
        if not self.steps:
            raise ValueError("No procedure loaded — cannot start")

        first = self.steps[0]
        self.state = FSMState(
            status="IN_PROGRESS",
            current_step_id=first.step_id,
            current_step_index=0,
            expected_action=first.action,
            expected_object=first.object,
            step_start_time=time.time(),
            next_instruction=first.description,
        )
        self.deviation_count = 0
        self._reset_debounce()
        print(f"[FSM] Procedure started: {self.procedure_name}")
        return {
            "event_type": "procedure_started",
            "state": self.state.to_dict(),
            "first_step": first.step_id,
            "instruction": first.description,
        }

    def process_observation(
        self,
        detected_action: str,
        detected_object: str,
        confidence: float,
        held_objects: list[str] | None = None,
    ) -> dict:
        """Process a single frame's observation.

        Called every frame (~30 times/second).

        Args:
            detected_action: Inferred action (PICK, OPEN, TRANSFER, CLOSE).
            detected_object: YOLO class name of the interacted object.
            confidence: Detection confidence [0.0, 1.0].
            held_objects: List of object class names currently in HELD state.

        Returns:
            dict with event_type and current state.
        """
        if self.state.status in ("IDLE", "COMPLETED"):
            state_dict = self.state.to_dict()
            state_dict["total_steps"] = self.total_steps
            return {"event_type": "none", "state": state_dict}

        current_step = self.steps[self.state.current_step_index]

        # --- CHECK TIMEOUT ---
        elapsed = time.time() - self.state.step_start_time
        if elapsed > current_step.timeout_seconds:
            return self._trigger_deviation(
                "INCOMPLETE_ACTION",
                f"Step {current_step.step_id} timed out after {elapsed:.0f}s. "
                f"Expected: {current_step.action} {current_step.object}",
            )

        # --- CHECK FOR CORRECT ACTION ---
        if (
            detected_action == current_step.action
            and detected_object == current_step.object
            and confidence >= current_step.confidence_threshold
        ):
            # Debounce: count consecutive correct detections
            if (
                self._debounce_action == detected_action
                and self._debounce_object == detected_object
            ):
                self._debounce_counter += 1
            else:
                self._debounce_action = detected_action
                self._debounce_object = detected_object
                self._debounce_counter = 1

            if self._debounce_counter >= self.debounce_frames:
                return self._advance_step(confidence)

            # Still debouncing
            self.state.confidence = confidence
            state_dict = self.state.to_dict()
            state_dict["total_steps"] = self.total_steps
            return {
                "event_type": "debouncing",
                "state": state_dict,
                "progress": f"{self._debounce_counter}/{self.debounce_frames}",
            }

        # --- CHECK FOR WRONG OBJECT ---
        if (
            detected_action == current_step.action
            and detected_object != current_step.object
            and detected_object != ""
            and confidence >= 0.5
        ):
            self._reset_debounce()
            return self._trigger_deviation(
                "WRONG_OBJECT",
                f"{detected_object} used instead of {current_step.object}. "
                f"Please use the correct object: {current_step.object}.",
            )

        # --- CHECK FOR SKIPPED STEP ---
        if detected_action and detected_object:
            for future_idx in range(self.state.current_step_index + 1, len(self.steps)):
                future_step = self.steps[future_idx]
                if (
                    detected_action == future_step.action
                    and detected_object == future_step.object
                    and confidence >= 0.5
                ):
                    skipped_ids = [
                        self.steps[i].step_id
                        for i in range(self.state.current_step_index, future_idx)
                    ]
                    self._reset_debounce()
                    return self._trigger_deviation(
                        "SKIPPED_STEP",
                        f"Steps {', '.join(skipped_ids)} were skipped. "
                        f"Please complete them in order.",
                    )

        # --- NO RELEVANT ACTION DETECTED ---
        self._reset_debounce()
        state_dict = self.state.to_dict()
        state_dict["total_steps"] = self.total_steps
        return {"event_type": "waiting", "state": state_dict}

    def _advance_step(self, confidence: float) -> dict:
        """Transition to the next step after debounce confirmation."""
        completed_step = self.steps[self.state.current_step_index]
        self.state.completed_steps.append(completed_step.step_id)
        self.state.deviation_type = None
        self.state.deviation_details = ""
        self.state.recovery_text = ""
        self.state.confidence = confidence

        next_idx = self.state.current_step_index + 1

        if next_idx >= len(self.steps):
            # Procedure complete!
            self.state.status = "COMPLETED"
            self.state.current_step_id = None
            self.state.expected_action = ""
            self.state.expected_object = ""
            self.state.next_instruction = "Experiment complete. All steps verified."
            self._reset_debounce()

            state_dict = self.state.to_dict()
            state_dict["total_steps"] = self.total_steps
            print(f"[FSM] ✅ Procedure COMPLETED")
            return {
                "event_type": "procedure_completed",
                "state": state_dict,
                "completed_step": completed_step.step_id,
            }

        # Move to next step
        next_step = self.steps[next_idx]
        self.state.current_step_index = next_idx
        self.state.current_step_id = next_step.step_id
        self.state.expected_action = next_step.action
        self.state.expected_object = next_step.object
        self.state.step_start_time = time.time()
        self.state.next_instruction = next_step.description
        self.state.confidence = 0.0
        self.state.status = "IN_PROGRESS"
        self._reset_debounce()

        state_dict = self.state.to_dict()
        state_dict["total_steps"] = self.total_steps
        print(
            f"[FSM] ✅ Step {completed_step.step_id} confirmed → "
            f"Now expecting {next_step.step_id}: {next_step.action} {next_step.object}"
        )
        return {
            "event_type": "step_confirmed",
            "state": state_dict,
            "completed_step": completed_step.step_id,
            "next_step": next_step.step_id,
        }

    def _trigger_deviation(self, dev_type: str, details: str) -> dict:
        """Record a deviation event."""
        self.state.status = "DEVIATION"
        self.state.deviation_type = dev_type
        self.state.deviation_details = details
        self.deviation_count += 1

        state_dict = self.state.to_dict()
        state_dict["total_steps"] = self.total_steps
        print(f"[FSM] ⚠️  DEVIATION [{dev_type}]: {details}")
        return {
            "event_type": "deviation_detected",
            "state": state_dict,
            "deviation_type": dev_type,
            "details": details,
        }

    def acknowledge_deviation(self):
        """Called when the astronaut corrects their action. Resume tracking."""
        self.state.status = "IN_PROGRESS"
        self.state.deviation_type = None
        self.state.deviation_details = ""
        self.state.recovery_text = ""
        self.state.step_start_time = time.time()  # Reset timeout
        self._reset_debounce()
        print("[FSM] Deviation acknowledged. Resuming procedure.")

    def get_current_step(self) -> StepDefinition | None:
        """Get the current step definition, or None if not in progress."""
        if 0 <= self.state.current_step_index < len(self.steps):
            return self.steps[self.state.current_step_index]
        return None

    def _reset_debounce(self):
        self._debounce_action = ""
        self._debounce_object = ""
        self._debounce_counter = 0

    def update_tool_state(self, tool_id: str, new_state: str):
        """Phase 19: Dynamically flip the state of a hazardous tool (OPEN/CLOSED)."""
        if tool_id in self.hazardous_containers:
            self.hazardous_containers[tool_id] = new_state

    def check_containment_breach(self, hand_z_coordinate: float) -> str:
        """Phase 19: Checks if hands left the rack while a hazardous tool is unsealed."""
        if hand_z_coordinate > self.settings.rack_boundary_z_max:
            for tool, state in self.hazardous_containers.items():
                if state == "OPEN":
                    return f"CRITICAL_HAZARD: {tool} UNSEALED"
        return "CONTAINMENT_SECURE"

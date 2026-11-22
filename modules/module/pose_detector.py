
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
import subprocess
from pathlib import Path
from datetime import datetime

from form_analyzer import analyze_squat_form
from database import initialize_database, save_session


# ============================================================
# K-FITNESS SETTINGS
# ============================================================

CAMERA_URL = "http://10.30.79.120:8080/video"

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"

WINDOW_NAME = "K-Fitness Live AI Voice Coach"

PROCESS_WIDTH = 800
MIN_VISIBILITY = 0.35

PANEL_W = 330
PANEL_H = 235

# Calibrated squat thresholds
STANDING_THRESHOLD = 138
BOTTOM_THRESHOLD = 125
DESCEND_THRESHOLD = 132

MIN_BOTTOM_FRAMES = 3
MIN_STANDING_FRAMES = 4
MIN_REP_INTERVAL = 1.0

FORM_VOICE_COOLDOWN = 8.0


# ============================================================
# CAMERA SELECTION
# ============================================================

def select_camera():

    print("\n")
    print("==========================================")
    print("       K-FITNESS CAMERA SELECTION")
    print("==========================================")
    print("1. Laptop Camera (Built-in Webcam)")
    print("2. IP Camera (Android Phone)")
    print("==========================================")

    while True:

        choice = input("Choose camera (1 or 2): ").strip()

        if choice == "1":

            print("[CAMERA] Laptop camera selected.")
            return "laptop"

        elif choice == "2":

            print("[CAMERA] IP camera selected.")
            return "ip"

        else:

            print("Invalid choice. Please enter 1 or 2.")


def open_camera(source):

    if source == "laptop":

        print("[CAMERA] Opening laptop webcam...")

        cap = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )

    elif source == "ip":

        print("[CAMERA] Opening IP camera:", CAMERA_URL)

        cap = cv2.VideoCapture(
            CAMERA_URL,
            cv2.CAP_FFMPEG
        )

    else:

        raise ValueError("Invalid camera source.")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():

        print("[CAMERA] Could not open selected camera.")

        cap.release()
        return None

    for _ in range(20):

        ok, frame = cap.read()

        if ok and frame is not None:

            print("[CAMERA] Connected successfully.")

            return cap

        time.sleep(0.15)

    print("[CAMERA] Camera did not provide frames.")

    cap.release()

    return None


# ============================================================
# OFFLINE VOICE COACH - WINDOWS SYSTEM.SPEECH
# ============================================================

class VoiceCoach:

    """Offline Windows voice using a persistent PowerShell process."""

    def __init__(self, pause_seconds=2.0):

        self.pause_seconds = pause_seconds

        self.q = queue.Queue(maxsize=1)

        self.ready = threading.Event()
        self.failed = threading.Event()

        self.running = True
        self.process = None

        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True
        )

        self.thread.start()

    def _worker(self):

        ps_script = (
            "$ErrorActionPreference='Stop'; "
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Volume = 100; $speaker.Rate = 0; "
            "[Console]::Out.WriteLine('KFITNESS_READY'); "
            "[Console]::Out.Flush(); "
            "while (($line = [Console]::In.ReadLine()) -ne $null) { "
            "if ($line -eq '__EXIT__') { break }; "
            "if ($line.Length -gt 0) { "
            "$speaker.Speak($line); "
            "[Console]::Out.WriteLine('KFITNESS_DONE'); "
            "[Console]::Out.Flush(); "
            "} }; $speaker.Dispose()"
        )

        try:

            self.process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    ps_script
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(
                    subprocess,
                    "CREATE_NO_WINDOW",
                    0
                )
            )

            while self.running:

                line = self.process.stdout.readline()

                if not line:

                    raise RuntimeError(
                        "PowerShell speech process closed before ready."
                    )

                if line.strip() == "KFITNESS_READY":

                    break

                print("[VOICE POWERSHELL]", line.strip())

            if not self.running:
                return

            self.ready.set()

            print("[VOICE] Windows System.Speech ready.")

            while self.running:

                message = self.q.get()

                try:

                    if message is None:
                        break

                    if self.process.poll() is not None:

                        raise RuntimeError(
                            "PowerShell voice process stopped."
                        )

                    clean = " ".join(
                        str(message).split()
                    ).replace("'", "")

                    print("[VOICE SPEAKING]", clean)

                    self.process.stdin.write(clean + "\n")
                    self.process.stdin.flush()

                    completed = False

                    while self.running:

                        line = self.process.stdout.readline()

                        if not line:
                            break

                        if line.strip() == "KFITNESS_DONE":

                            completed = True
                            break

                        if line.strip():

                            print(
                                "[VOICE POWERSHELL]",
                                line.strip()
                            )

                    if completed and self.running:

                        time.sleep(self.pause_seconds)

                except Exception as exc:

                    self.failed.set()

                    print(
                        "[VOICE PLAYBACK ERROR]",
                        repr(exc)
                    )

                finally:

                    self.q.task_done()

        except Exception as exc:

            self.failed.set()

            print("[VOICE ERROR]", repr(exc))

        finally:

            self.ready.clear()

            if self.process is not None:

                try:

                    if self.process.poll() is None:

                        self.process.stdin.write("__EXIT__\n")
                        self.process.stdin.flush()

                        self.process.wait(timeout=2)

                except Exception:

                    try:
                        self.process.terminate()
                    except Exception:
                        pass

    def speak(self, message, priority=False):

        if not self.running or not self.ready.is_set():
            return

        with self.lock:

            # Avoid creating a queue of outdated voice instructions.
            if not self.q.empty():
                return

            try:

                self.q.put_nowait(message)

            except queue.Full:
                pass

    def stop(self):

        self.running = False

        try:

            while True:

                self.q.get_nowait()
                self.q.task_done()

        except queue.Empty:
            pass

        try:

            self.q.put_nowait(None)

        except queue.Full:
            pass


# ============================================================
# STABLE SQUAT COUNTER
# ============================================================

class StableSquatCounter:

    """Counts only after a valid descent and return to standing."""

    def __init__(self):

        self.reps = 0
        self.phase = "WAITING"

        self.smoothed = None
        self.alpha = 0.30

        self.bottom_frames = 0
        self.standing_frames = 0

        self.last_rep_time = 0.0
        self.last_angle = None

    def update(self, raw_angle):

        if raw_angle is None:

            return self.result(
                None,
                "Keep hip, knee and ankle visible"
            )

        if not 35 <= raw_angle <= 180:

            return self.result(
                self.smoothed,
                "Turn sideways and keep your legs visible"
            )

        if self.smoothed is None:

            self.smoothed = raw_angle

        else:

            self.smoothed = (
                self.alpha * raw_angle
                + (1 - self.alpha) * self.smoothed
            )

        angle = self.smoothed

        feedback = "Move naturally"

        if self.phase == "WAITING":

            if angle >= STANDING_THRESHOLD:

                self.standing_frames += 1

            else:

                self.standing_frames = 0

            if self.standing_frames >= MIN_STANDING_FRAMES:

                self.phase = "STANDING"
                self.standing_frames = 0

            feedback = "Stand upright to begin"

        elif self.phase == "STANDING":

            if angle <= DESCEND_THRESHOLD:

                self.phase = "DESCENDING"
                self.bottom_frames = 0

                feedback = "Lower yourself slowly"

            else:

                feedback = "Good posture. Prepare for your next squat"

        elif self.phase == "DESCENDING":

            if angle <= BOTTOM_THRESHOLD:

                self.bottom_frames += 1

                if self.bottom_frames >= MIN_BOTTOM_FRAMES:

                    self.phase = "BOTTOM"

                    feedback = "Good depth. Push upward steadily"

                else:

                    feedback = "A little lower and hold your position"

            else:

                self.bottom_frames = 0

                if angle >= STANDING_THRESHOLD:

                    self.phase = "STANDING"

                    feedback = "Squat lower to complete a repetition"

                else:

                    feedback = "Continue lowering slowly"

        elif self.phase == "BOTTOM":

            if angle >= DESCEND_THRESHOLD:

                self.phase = "ASCENDING"
                self.standing_frames = 0

                feedback = "Stand up steadily"

            else:

                feedback = "Push upward steadily"

        elif self.phase == "ASCENDING":

            if angle >= STANDING_THRESHOLD:

                self.standing_frames += 1

                if self.standing_frames >= MIN_STANDING_FRAMES:

                    now = time.monotonic()

                    if now - self.last_rep_time >= MIN_REP_INTERVAL:

                        self.reps += 1
                        self.last_rep_time = now

                    self.phase = "STANDING"

                    self.standing_frames = 0
                    self.bottom_frames = 0

                    feedback = "Repetition completed. Great work"

                else:

                    feedback = "Almost upright. Finish standing"

            else:

                self.standing_frames = 0

                feedback = "Keep standing up steadily"

        self.last_angle = angle

        return self.result(angle, feedback)

    def result(self, angle, feedback):

        return {
            "reps": self.reps,
            "phase": self.phase,
            "angle": angle,
            "feedback": feedback
        }


# ============================================================
# MEDIAPIPE CONFIGURATION
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Pose model not found: {MODEL_PATH}. "
        "Put pose_landmarker_lite.task in the project root."
    )

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker

PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

RunningMode = mp.tasks.vision.RunningMode

POSE_OPTIONS = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),

    running_mode=RunningMode.VIDEO,

    num_poses=1,

    min_pose_detection_confidence=0.5,

    min_pose_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


CONNECTIONS = [
    (11,12), (11,13), (13,15), (12,14),
    (14,16), (11,23), (12,24), (23,24),
    (23,25), (25,27), (24,26), (26,28),
    (27,29), (29,31), (28,30), (30,32)
]


# ============================================================
# LANDMARK HELPERS
# ============================================================

def visible(lm):

    return (
        getattr(lm, "visibility", 0.0) >= MIN_VISIBILITY
        and 0 <= lm.x <= 1
        and 0 <= lm.y <= 1
    )


def angle_3pt(a, b, c):

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)

    ba = a - b
    bc = c - b

    denom = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denom < 1e-8:
        return None

    cosine = np.clip(
        np.dot(ba, bc) / denom,
        -1,
        1
    )

    return float(
        np.degrees(np.arccos(cosine))
    )


def knee_angle_for_side(lms, ids, w, h):

    hip, knee, ankle = (
        lms[i] for i in ids
    )

    if not all(
        visible(p)
        for p in (hip, knee, ankle)
    ):

        return None, 0.0

    pts = [
        (p.x * w, p.y * h)
        for p in (hip, knee, ankle)
    ]

    confidence = min(
        getattr(p, "visibility", 0.0)
        for p in (hip, knee, ankle)
    )

    return angle_3pt(*pts), confidence


def choose_knee_angle(lms, w, h):

    left = knee_angle_for_side(
        lms,
        (23,25,27),
        w,
        h
    )

    right = knee_angle_for_side(
        lms,
        (24,26,28),
        w,
        h
    )

    options = []

    if left[0] is not None:

        options.append(
            (left[0], left[1], "LEFT")
        )

    if right[0] is not None:

        options.append(
            (right[0], right[1], "RIGHT")
        )

    if not options:
        return None, None

    angle, _, side = max(
        options,
        key=lambda x: x[1]
    )

    return angle, side


# ============================================================
# DRAW SKELETON
# ============================================================

def draw_skeleton(frame, lms):

    h, w = frame.shape[:2]

    for a, b in CONNECTIONS:

        p1, p2 = lms[a], lms[b]

        if visible(p1) and visible(p2):

            cv2.line(
                frame,
                (int(p1.x*w), int(p1.y*h)),
                (int(p2.x*w), int(p2.y*h)),
                (0,230,0),
                2
            )

    for p in lms:

        if visible(p):

            cv2.circle(
                frame,
                (int(p.x*w), int(p.y*h)),
                4,
                (0,0,255),
                -1
            )


# ============================================================
# DASHBOARD
# ============================================================

def draw_dashboard(
    frame,
    fps,
    angle,
    side,
    reps,
    phase,
    feedback,
    voice_on,
    form_score=None,
    form_feedback="Waiting for analysis"
):

    h, w = frame.shape[:2]

    x, y = 10, 10

    pw = min(PANEL_W, w-20)
    ph = PANEL_H

    cv2.rectangle(
        frame,
        (x,y),
        (x+pw,y+ph),
        (18,18,18),
        -1
    )

    def put(text, yy, color=(245,245,245), scale=0.34):

        text = str(text)

        max_chars = max(
            16,
            int((pw-20)/(8.0*scale))
        )

        if len(text) > max_chars:

            text = text[:max_chars-3] + "..."

        cv2.putText(
            frame,
            text,
            (x+9,yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA
        )

    put(
        "K-FITNESS LIVE COACH",
        y+22,
        (0,255,255),
        0.40
    )

    put(
        f"FPS: {fps:.1f} | VOICE: {'ON' if voice_on else 'OFF'}",
        y+44,
        (0,255,0),
        0.32
    )

    put(
        f"Knee: {angle:.1f} deg"
        if angle is not None
        else "Knee: not detected",
        y+66,
        (255,255,0),
        0.34
    )

    put(
        f"Leg: {side or 'NONE'}",
        y+88,
        (245,245,245),
        0.34
    )

    put(
        f"Squat reps: {reps}",
        y+110,
        (0,255,0),
        0.38
    )

    put(
        f"Phase: {phase}",
        y+132,
        (245,245,245),
        0.34
    )

    score_text = (
        f"Form score: {form_score}/100"
        if form_score is not None
        else "Form score: --"
    )

    score_color = (
        (0,255,0)
        if form_score is not None and form_score >= 80
        else (0,200,255)
    )

    put(
        score_text,
        y+154,
        score_color,
        0.36
    )

    put(
        "Rep: " + str(feedback),
        y+176,
        (0,200,255),
        0.29
    )

    put(
        "Form: " + str(form_feedback),
        y+198,
        (0,200,255),
        0.29
    )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nK-FITNESS - AI MOVEMENT COACH")

    print("Running:", Path(__file__).resolve())

    print("Model:", MODEL_PATH)

    # Ask which camera to use before starting.
    camera_source = select_camera()

    voice = VoiceCoach(
        pause_seconds=2.0
    )

    if voice.ready.wait(timeout=10):

        voice.speak(
            "Welcome to K Fitness. "
            "Stand sideways and keep your full body visible."
        )

    else:

        print(
            "[WARNING] Voice did not initialize. "
            "Check Windows System.Speech."
        )

    cap = open_camera(camera_source)

    if cap is None:

        voice.stop()

        raise RuntimeError(
            "Selected camera could not be opened. "
            "Check camera permissions or IP camera connection."
        )

    # --------------------------------------------------------
    # LOCAL WORKOUT SESSION TRACKING (SQLite)
    # --------------------------------------------------------
    initialize_database()

    session_started_at = datetime.now()
    session_start_time = time.monotonic()

    score_total = 0.0
    score_count = 0
    best_score = None
    lowest_score = None

    form_corrections = 0
    last_correction = None
    last_correction_time = 0.0

    counter = StableSquatCounter()

    last_spoken_phase = None
    last_rep_announced = 0

    last_frame_time = time.time()

    fps = 0.0
    timestamp_ms = 0

    last_valid_angle_time = time.monotonic()

    last_form_voice_time = 0.0
    last_form_voice_message = None

    form_score = None
    form_feedback = "Waiting for analysis"

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        1050,
        650
    )

    prompts = {

        "WAITING":
            "Stand upright to begin.",

        "STANDING":
            "Good posture. Prepare for your next squat.",

        "DESCENDING":
            "Lower yourself slowly. Keep your movement controlled.",

        "BOTTOM":
            "Good depth. Push upward steadily.",

        "ASCENDING":
            "Stand up steadily and straighten your legs."
    }

    try:

        with PoseLandmarker.create_from_options(
            POSE_OPTIONS
        ) as landmarker:

            while True:

                ok, frame = cap.read()

                if not ok or frame is None:

                    print(
                        "[CAMERA] Frame lost; reconnecting..."
                    )

                    cap.release()

                    time.sleep(0.8)

                    # Reconnect to the SAME selected camera.
                    cap = open_camera(camera_source)

                    if cap is None:
                        break

                    continue

                # --------------------------------------------
                # RESIZE AND MIRROR
                # --------------------------------------------

                h0, w0 = frame.shape[:2]

                if w0 > PROCESS_WIDTH:

                    new_h = int(
                        h0 * PROCESS_WIDTH / w0
                    )

                    frame = cv2.resize(
                        frame,
                        (PROCESS_WIDTH,new_h),
                        interpolation=cv2.INTER_AREA
                    )

                frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]

                # --------------------------------------------
                # MEDIAPIPE
                # --------------------------------------------

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = max(
                    timestamp_ms+1,
                    int(time.monotonic()*1000)
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                # --------------------------------------------
                # FPS
                # --------------------------------------------

                now = time.time()

                instant_fps = 1.0 / max(
                    now-last_frame_time,
                    0.001
                )

                fps = (
                    instant_fps
                    if fps == 0
                    else 0.85*fps + 0.15*instant_fps
                )

                last_frame_time = now

                # --------------------------------------------
                # DEFAULT VALUES
                # --------------------------------------------

                angle, side = None, None

                feedback = "Move into camera view"

                reps = counter.reps
                phase = counter.phase

                # --------------------------------------------
                # POSE DETECTED
                # --------------------------------------------

                if result.pose_landmarks:

                    lms = result.pose_landmarks[0]

                    draw_skeleton(
                        frame,
                        lms
                    )

                    # ----------------------------------------
                    # FORM ANALYSIS
                    # ----------------------------------------

                    try:

                        form_result = analyze_squat_form(
                            lms,
                            w,
                            h
                        )

                        form_score = form_result["score"]

                        form_feedback = form_result["feedback"]

                        # Record valid form scores for this session.
                        current_score = form_result.get("score")
                        if isinstance(current_score, (int, float)):
                            score_value = float(current_score)
                            score_total += score_value
                            score_count += 1

                            best_score = (
                                score_value if best_score is None
                                else max(best_score, score_value)
                            )
                            lowest_score = (
                                score_value if lowest_score is None
                                else min(lowest_score, score_value)
                            )

                        # Count distinct correction events, not every frame.
                        if form_result.get("status") == "NEEDS_ATTENTION":
                            current_correction = str(form_feedback).strip()
                            current_time = time.monotonic()

                            if current_correction and (
                                current_correction != last_correction
                                or current_time - last_correction_time >= 5.0
                            ):
                                form_corrections += 1
                                last_correction = current_correction
                                last_correction_time = current_time
                        else:
                            last_correction = None

                    except Exception as exc:

                        print(
                            "[FORM ANALYZER ERROR]",
                            repr(exc)
                        )

                        form_score = None

                        form_feedback = "Form analysis unavailable"

                        form_result = {
                            "status": "INSUFFICIENT_DATA",
                            "feedback": form_feedback
                        }

                    # ----------------------------------------
                    # KNEE ANGLE
                    # ----------------------------------------

                    angle, side = choose_knee_angle(
                        lms,
                        w,
                        h
                    )

                    if angle is not None:

                        last_valid_angle_time = time.monotonic()

                        data = counter.update(angle)

                        reps = data["reps"]

                        phase = data["phase"]

                        feedback = data["feedback"]

                        # ------------------------------------
                        # PHASE VOICE FEEDBACK
                        # ------------------------------------

                        if phase != last_spoken_phase:

                            if phase in prompts:

                                voice.speak(
                                    prompts[phase]
                                )

                            last_spoken_phase = phase

                        # ------------------------------------
                        # REP ANNOUNCEMENT
                        # ------------------------------------

                        if reps > last_rep_announced:

                            voice.speak(
                                f"Squat {reps} completed. Well done!"
                            )

                            last_rep_announced = reps

                        # ------------------------------------
                        # FORM CORRECTION VOICE
                        # ------------------------------------

                        active_phase = phase in (
                            "DESCENDING",
                            "BOTTOM",
                            "ASCENDING"
                        )

                        if (
                            active_phase
                            and form_result["status"] == "NEEDS_ATTENTION"
                            and form_feedback
                            and form_feedback != last_form_voice_message
                        ):

                            current_time = time.monotonic()

                            if (
                                current_time - last_form_voice_time
                                >= FORM_VOICE_COOLDOWN
                            ):

                                voice.speak(
                                    form_feedback
                                )

                                last_form_voice_time = current_time

                                last_form_voice_message = form_feedback

                    else:

                        feedback = (
                            "Keep hip, knee and ankle visible"
                        )

                        if (
                            time.monotonic()
                            - last_valid_angle_time > 5
                        ):

                            voice.speak(
                                "Please step back and keep your "
                                "hip, knee and ankle visible."
                            )

                # --------------------------------------------
                # PERSON NOT DETECTED
                # --------------------------------------------

                else:

                    feedback = "Person not detected"

                    if (
                        time.monotonic()
                        - last_valid_angle_time > 5
                    ):

                        voice.speak(
                            "Please move into the camera frame."
                        )

                # --------------------------------------------
                # DASHBOARD
                # --------------------------------------------

                draw_dashboard(
                    frame,
                    fps,
                    angle,
                    side,
                    reps,
                    phase,
                    feedback,
                    voice.ready.is_set(),
                    form_score,
                    form_feedback
                )

                cv2.imshow(
                    WINDOW_NAME,
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), 27):
                    break

    finally:

        # Save the workout even if the user exits with Q or the loop stops.
        try:
            ended_at = datetime.now()
            duration_seconds = time.monotonic() - session_start_time

            average_score = (
                score_total / score_count
                if score_count > 0
                else None
            )

            save_session(
                started_at=session_started_at.isoformat(timespec="seconds"),
                ended_at=ended_at.isoformat(timespec="seconds"),
                duration_seconds=round(duration_seconds, 2),
                exercise="Squats",
                reps=counter.reps,
                avg_form_score=(
                    round(average_score, 2)
                    if average_score is not None
                    else None
                ),
                best_form_score=(
                    round(best_score)
                    if best_score is not None
                    else None
                ),
                lowest_form_score=(
                    round(lowest_score)
                    if lowest_score is not None
                    else None
                ),
                form_corrections=form_corrections
            )

            print("[DATABASE] Workout session saved successfully.")
            print(f"[DATABASE] Repetitions: {counter.reps}")
            print(f"[DATABASE] Average form score: {average_score}")

        except Exception as db_error:
            print("[DATABASE SAVE ERROR]", repr(db_error))

        cap.release()

        cv2.destroyAllWindows()

        voice.stop()

        print("K-Fitness stopped.")


if __name__ == "__main__":
    main()

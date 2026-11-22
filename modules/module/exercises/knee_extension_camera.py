
import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import queue
import subprocess

from pathlib import Path

from knee_extension import KneeExtensionCounter
from knee_extension_analyzer import KneeExtensionAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_URL = "http://10.30.79.120:8080/video"

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"

WINDOW_NAME = "K-Fitness - Knee Extension Coach"

PROCESS_WIDTH = 1280
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 850

MIN_VISIBILITY = 0.40

# Knee extension calibration
BENT_THRESHOLD = 100
EXTENDED_THRESHOLD = 120

MIN_HOLD_FRAMES = 3
MIN_REP_INTERVAL = 1.0

# MediaPipe landmarks
LEFT_IDS = (23, 25, 27)
RIGHT_IDS = (24, 26, 28)

# Full body skeleton connections
CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31),
    (28, 30), (30, 32),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22)
]


# ============================================================
# VOICE COACH
# ============================================================

class VoiceCoach:

    def __init__(self, pause_seconds=2.0):

        self.pause_seconds = pause_seconds
        self.q = queue.Queue(maxsize=1)
        self.ready = threading.Event()
        self.running = True
        self.process = None
        self.last_spoken = {}
        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True
        )
        self.thread.start()

    def _worker(self):

        script = (
            "$ErrorActionPreference='Stop'; "
            "Add-Type -AssemblyName System.Speech; "
            "$speaker=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Volume=100; $speaker.Rate=0; "
            "[Console]::Out.WriteLine('READY'); "
            "[Console]::Out.Flush(); "
            "while (($line=[Console]::In.ReadLine()) -ne $null) { "
            "if ($line -eq '__EXIT__') { break }; "
            "if ($line.Length -gt 0) { "
            "$speaker.Speak($line); "
            "[Console]::Out.WriteLine('DONE'); "
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
                    script
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
                    raise RuntimeError("Voice process stopped.")

                if line.strip() == "READY":
                    break

            self.ready.set()
            print("[VOICE] System.Speech ready.")

            while self.running:

                message = self.q.get()

                try:

                    if message is None:
                        break

                    if self.process.poll() is not None:
                        break

                    self.process.stdin.write(message + "\n")
                    self.process.stdin.flush()

                    while self.running:

                        line = self.process.stdout.readline()

                        if not line:
                            break

                        if line.strip() == "DONE":
                            break

                    if self.running:
                        time.sleep(self.pause_seconds)

                finally:
                    self.q.task_done()

        except Exception as exc:
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
                    self.process.terminate()

    def speak(self, message, cooldown=5):

        if not self.running or not self.ready.is_set():
            return

        now = time.monotonic()

        with self.lock:

            previous = self.last_spoken.get(message, 0)

            if now - previous < cooldown:
                return

            if self.q.empty():

                try:
                    self.q.put_nowait(message)
                    self.last_spoken[message] = now
                except queue.Full:
                    pass

    def stop(self):

        self.running = False

        try:
            self.q.put_nowait(None)
        except queue.Full:
            pass


# ============================================================
# CAMERA
# ============================================================

def open_camera(choice):

    if choice == "1":

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    elif choice == "2":

        cap = cv2.VideoCapture(
            CAMERA_URL,
            cv2.CAP_FFMPEG
        )

    else:
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        cap.release()
        return None

    for _ in range(20):

        ok, frame = cap.read()

        if ok and frame is not None:
            return cap

        time.sleep(0.1)

    cap.release()
    return None


# ============================================================
# ANGLE CALCULATION
# ============================================================

def calculate_angle(a, b, c):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return None

    cosine = np.clip(
        np.dot(ba, bc) / denominator,
        -1,
        1
    )

    return float(np.degrees(np.arccos(cosine)))


# ============================================================
# SELECT BEST VISIBLE LEG
# ============================================================

def choose_leg(landmarks, width, height):

    candidates = []

    for side, ids in [
        ("LEFT", LEFT_IDS),
        ("RIGHT", RIGHT_IDS)
    ]:

        points = [landmarks[i] for i in ids]

        visibility = [
            getattr(p, "visibility", 0)
            for p in points
        ]

        if min(visibility) < MIN_VISIBILITY:
            continue

        coordinates = [
            (p.x * width, p.y * height)
            for p in points
        ]

        angle = calculate_angle(*coordinates)

        if angle is not None:

            candidates.append((
                min(visibility),
                angle,
                side,
                coordinates
            ))

    if not candidates:
        return None, None, None, None

    return max(candidates, key=lambda x: x[0])


# ============================================================
# IMPROVED REP COUNTER
# ============================================================

class KneeExtensionCounter:

    def __init__(self):

        self.reps = 0
        self.phase = "WAITING"

        self.smoothed_angle = None
        self.alpha = 0.35

        self.bent_frames = 0
        self.extended_frames = 0

        self.last_rep_time = 0

    def update(self, raw_angle):

        if raw_angle is None:
            return self.result(None, "Keep your leg visible")

        if not 35 <= raw_angle <= 180:
            return self.result(
                self.smoothed_angle,
                "Adjust your sitting position"
            )

        if self.smoothed_angle is None:

            self.smoothed_angle = raw_angle

        else:

            self.smoothed_angle = (
                self.alpha * raw_angle
                + (1 - self.alpha) * self.smoothed_angle
            )

        angle = self.smoothed_angle

        feedback = "Move slowly and steadily"

        # START POSITION
        if self.phase == "WAITING":

            if angle <= BENT_THRESHOLD:
                self.bent_frames += 1
            else:
                self.bent_frames = 0

            if self.bent_frames >= MIN_HOLD_FRAMES:

                self.phase = "BENT"
                self.bent_frames = 0

                feedback = "Starting position detected. Extend your knee."

            else:
                feedback = "Bend your knee comfortably to begin"

        # BENT TO EXTENDED
        elif self.phase == "BENT":

            if angle >= EXTENDED_THRESHOLD:

                self.extended_frames += 1
                feedback = "Hold your leg extended"

                if self.extended_frames >= MIN_HOLD_FRAMES:

                    self.phase = "EXTENDED"
                    self.extended_frames = 0

                    feedback = "Good extension. Slowly bend your knee."

            else:

                self.extended_frames = 0
                feedback = "Slowly straighten your knee"

        # EXTENDED TO BENT = ONE REP
        elif self.phase == "EXTENDED":

            if angle <= BENT_THRESHOLD:

                self.bent_frames += 1
                feedback = "Return your knee slowly"

                if self.bent_frames >= MIN_HOLD_FRAMES:

                    now = time.monotonic()

                    if now - self.last_rep_time >= MIN_REP_INTERVAL:

                        self.reps += 1
                        self.last_rep_time = now

                    self.phase = "BENT"
                    self.bent_frames = 0

                    feedback = "Repetition completed. Great work!"

            else:

                self.bent_frames = 0
                feedback = "Continue returning your knee"

        return self.result(angle, feedback)

    def result(self, angle, feedback):

        return {
            "reps": self.reps,
            "phase": self.phase,
            "angle": round(angle, 1) if angle is not None else None,
            "feedback": feedback
        }


# ============================================================
# DRAW FULL SKELETON
# ============================================================

def draw_skeleton(frame, landmarks):

    h, w = frame.shape[:2]

    for a, b in CONNECTIONS:

        p1 = landmarks[a]
        p2 = landmarks[b]

        if (
            getattr(p1, "visibility", 0) >= MIN_VISIBILITY
            and getattr(p2, "visibility", 0) >= MIN_VISIBILITY
        ):

            cv2.line(
                frame,
                (int(p1.x * w), int(p1.y * h)),
                (int(p2.x * w), int(p2.y * h)),
                (0, 255, 0),
                3
            )

    for point in landmarks:

        if getattr(point, "visibility", 0) >= MIN_VISIBILITY:

            cv2.circle(
                frame,
                (int(point.x * w), int(point.y * h)),
                4,
                (0, 0, 255),
                -1
            )


# ============================================================
# DASHBOARD
# ============================================================

def draw_dashboard(
    frame,
    angle,
    reps,
    phase,
    score,
    feedback,
    side,
    voice_on,
    fps
):

    h, w = frame.shape[:2]

    panel_w = min(460, w - 20)

    cv2.rectangle(
        frame,
        (10, 10),
        (panel_w, 275),
        (20, 20, 20),
        -1
    )

    lines = [
        ("K-FITNESS - KNEE EXTENSION", (0, 255, 255)),
        (f"FPS: {fps:.1f} | VOICE: {'ON' if voice_on else 'OFF'}", (0, 255, 0)),
        (f"Selected leg: {side or '--'}", (255, 255, 255)),
        (f"Knee angle: {angle:.1f}" if angle is not None else "Knee angle: --", (255, 255, 0)),
        (f"Repetitions: {reps}", (0, 255, 0)),
        (f"Phase: {phase}", (255, 255, 255)),
        (f"Form score: {score}/100" if score is not None else "Form score: --", (0, 200, 255)),
        ("Feedback: " + str(feedback)[:45], (0, 200, 255))
    ]

    for i, (text, color) in enumerate(lines):

        cv2.putText(
            frame,
            text,
            (25, 42 + i * 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n====================================")
    print(" K-FITNESS - KNEE EXTENSION COACH")
    print("====================================")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Pose model not found: {MODEL_PATH}"
        )

    print("\n1. Laptop Camera")
    print("2. IP Camera")

    choice = input("Choose camera (1 or 2): ").strip()

    cap = open_camera(choice)

    if cap is None:
        raise RuntimeError("Could not open selected camera.")

    counter = KneeExtensionCounter()
    analyzer = KneeExtensionAnalyzer()
    voice = VoiceCoach(pause_seconds=2.0)

    if voice.ready.wait(timeout=10):
        voice.speak(
            "Welcome to K Fitness. Sit sideways and keep your full leg visible."
        )

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    timestamp_ms = 0
    last_frame_time = time.monotonic()
    fps = 0.0

    last_announced_rep = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        WINDOW_NAME,
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT
    )

    try:

        with PoseLandmarker.create_from_options(options) as landmarker:

            while True:

                ok, frame = cap.read()

                if not ok or frame is None:
                    print("[CAMERA] Frame unavailable.")
                    break

                h, w = frame.shape[:2]

                if w > PROCESS_WIDTH:

                    new_h = int(h * PROCESS_WIDTH / w)

                    frame = cv2.resize(
                        frame,
                        (PROCESS_WIDTH, new_h),
                        interpolation=cv2.INTER_AREA
                    )

                frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = max(
                    timestamp_ms + 1,
                    int(time.monotonic() * 1000)
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                now = time.monotonic()

                instant_fps = 1.0 / max(
                    now - last_frame_time,
                    0.001
                )

                fps = (
                    instant_fps if fps == 0
                    else 0.85 * fps + 0.15 * instant_fps
                )

                last_frame_time = now

                angle = None
                side = None
                score = None
                feedback = "Person not detected"

                if result.pose_landmarks:

                    landmarks = result.pose_landmarks[0]

                    # Full body skeleton
                    draw_skeleton(frame, landmarks)

                    # Select visible leg and calculate angle
                    confidence, raw_angle, side, coordinates = choose_leg(
                        landmarks,
                        w,
                        h
                    )

                    if raw_angle is not None:

                        angle = raw_angle

                        data = counter.update(angle)

                        feedback = data["feedback"]

                        # Form analyzer
                        try:

                            form_result = analyzer.analyze(
                                landmarks,
                                w,
                                h
                            )

                            score = form_result.get("score")

                        except Exception as exc:

                            print("[ANALYZER ERROR]", repr(exc))

                        # Highlight selected leg
                        points = [
                            (int(p[0]), int(p[1]))
                            for p in coordinates
                        ]

                        cv2.line(
                            frame,
                            points[0],
                            points[1],
                            (255, 0, 255),
                            4
                        )

                        cv2.line(
                            frame,
                            points[1],
                            points[2],
                            (255, 0, 255),
                            4
                        )

                        for point in points:

                            cv2.circle(
                                frame,
                                point,
                                7,
                                (0, 0, 255),
                                -1
                            )

                        # Rep announcement
                        if counter.reps > last_announced_rep:

                            voice.speak(
                                f"Repetition {counter.reps} completed. Well done!"
                            )

                            last_announced_rep = counter.reps

                        if counter.phase == "EXTENDED":
                            voice.speak(
                                "Good extension. Slowly bend your knee."
                            )

                        # Form feedback
                        if (
                            isinstance(form_result, dict)
                            and form_result.get("status") == "NEEDS_ATTENTION"
                        ):
                            voice.speak(
                                form_result.get("feedback", ""),
                                cooldown=8
                            )

                    else:

                        feedback = "Keep hip, knee and ankle visible"

                # Dashboard
                draw_dashboard(
                    frame,
                    angle,
                    counter.reps,
                    counter.phase,
                    score,
                    feedback,
                    side,
                    voice.ready.is_set(),
                    fps
                )

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

    finally:

        cap.release()
        cv2.destroyAllWindows()
        voice.stop()

    print("\nKnee Extension Session Finished")
    print("Total repetitions:", counter.reps)


if __name__ == "__main__":
    main()

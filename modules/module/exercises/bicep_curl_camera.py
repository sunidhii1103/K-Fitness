
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import subprocess
from pathlib import Path
import sys

MODULE_DIR = Path(__file__).resolve().parents[1]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from session_tracker import WorkoutSessionTracker


# =====================================================
# PROJECT SETTINGS
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"

PHONE_CAMERA_URL = "http://10.30.79.120:8080/video"

DISPLAY_WIDTH = 1280
WINDOW_NAME = "K-Fitness - Bicep Curls"

EXTENDED_ANGLE = 150
CURLED_ANGLE = 65
MIN_REP_INTERVAL = 0.8


# =====================================================
# MEDIAPIPE
# =====================================================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# =====================================================
# VOICE COACH
# =====================================================

class VoiceCoach:

    def __init__(self):

        self.process = None
        self.last_message = ""
        self.last_time = 0

        try:

            powershell_script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Rate = 0; "
                "while (($line = [Console]::In.ReadLine()) -ne $null) { "
                "if ($line -eq '__QUIT__') { break }; "
                "if ($line.Length -gt 0) { $s.Speak($line) } "
                "}; $s.Dispose()"
            )

            self.process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy", "Bypass",
                    "-Command", powershell_script
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )

        except Exception as error:
            print("Voice engine could not start:", error)

    def speak(self, message, force=False):

        if self.process is None:
            return

        current_time = time.time()

        if not force:

            if message == self.last_message:
                return

            if current_time - self.last_time < 1.0:
                return

        try:

            self.process.stdin.write(message + "\n")
            self.process.stdin.flush()

            self.last_message = message
            self.last_time = current_time

        except Exception:
            pass

    def close(self):

        if self.process is not None:

            try:
                self.process.stdin.write("__QUIT__\n")
                self.process.stdin.flush()
                self.process.stdin.close()
                self.process.wait(timeout=3)

            except Exception:
                self.process.terminate()


# =====================================================
# ANGLE CALCULATION
# =====================================================

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = (
        math.atan2(c[1] - b[1], c[0] - b[0])
        - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(radians * 180.0 / math.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


# =====================================================
# BICEP CURL COUNTER
# =====================================================

class BicepCurlCounter:

    def __init__(self):

        self.reps = 0
        self.stage = "START"
        self.last_rep_time = 0
        self.feedback = "Extend your arm"

    def update(self, angle):

        current_time = time.time()

        if angle >= EXTENDED_ANGLE:

            self.stage = "EXTENDED"
            self.feedback = "Curl your arm"

        elif angle <= CURLED_ANGLE and self.stage == "EXTENDED":

            if current_time - self.last_rep_time >= MIN_REP_INTERVAL:

                self.reps += 1
                self.last_rep_time = current_time

                self.stage = "CURLED"
                self.feedback = "Good repetition!"

        elif self.stage == "CURLED":

            self.feedback = "Extend your arm again"

        return self.reps, self.feedback


# =====================================================
# CAMERA SELECTION
# =====================================================

def select_camera():

    print("\n" + "=" * 45)
    print("          SELECT YOUR CAMERA")
    print("=" * 45)

    print("1. Laptop Camera")
    print("2. Phone IP Camera")

    choice = input("\nEnter your choice (1/2): ").strip()

    if choice == "1":

        print("\nConnecting to laptop camera...")

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    elif choice == "2":

        print("\nConnecting to phone IP camera...")
        print("Using saved IP Webcam address.")

        cap = cv2.VideoCapture(PHONE_CAMERA_URL)

    else:

        print("\nInvalid camera selection.")
        return None

    if not cap.isOpened():

        print("\nCamera connection failed.")
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera connected successfully!")

    return cap


# =====================================================
# SKELETON CONNECTIONS
# =====================================================

CONNECTIONS = [

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (11, 23),
    (12, 24),

    (23, 24),

    (23, 25),
    (25, 27),

    (24, 26),
    (26, 28),

    (27, 29),
    (28, 30),

    (27, 31),
    (28, 32),

    (29, 31),
    (30, 32)
]


# =====================================================
# MAIN PROGRAM
# =====================================================

def main():

    if not MODEL_PATH.exists():

        print("\nPose model not found:")
        print(MODEL_PATH)
        return

    print("\n" + "=" * 45)
    print("          K-FITNESS BICEP CURLS")
    print("=" * 45)

    print("\nSelect the arm:")
    print("1. Left Arm")
    print("2. Right Arm")

    side_choice = input("\nEnter your choice (1/2): ").strip()

    if side_choice == "1":

        shoulder_id = 11
        elbow_id = 13
        wrist_id = 15
        arm_name = "LEFT"

    elif side_choice == "2":

        shoulder_id = 12
        elbow_id = 14
        wrist_id = 16
        arm_name = "RIGHT"

    else:

        print("Invalid arm selection.")
        return

    cap = select_camera()

    if cap is None:
        return

    counter = BicepCurlCounter()
    session = WorkoutSessionTracker("Bicep Curls")
    voice = VoiceCoach()

    start_time = time.time()
    frame_timestamp = 0

    previous_stage = counter.stage

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

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(
        WINDOW_NAME,
        1320,
        850
    )

    print("\nBicep Curl session started.")
    print("Voice feedback enabled.")
    print("Press Q to quit.")

    voice.speak(
        "Welcome to K Fitness. "
        "Bicep curl exercise started. "
        "Keep your selected arm visible.",
        force=True
    )

    try:

        with PoseLandmarker.create_from_options(options) as landmarker:

            while cap.isOpened():

                success, frame = cap.read()

                if not success:

                    print("Unable to read camera frame.")
                    break

                frame = cv2.flip(frame, 1)

                original_height, original_width = frame.shape[:2]

                scale = DISPLAY_WIDTH / original_width

                display_height = int(original_height * scale)

                frame = cv2.resize(
                    frame,
                    (DISPLAY_WIDTH, display_height),
                    interpolation=cv2.INTER_LINEAR
                )

                height, width = frame.shape[:2]

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                frame_timestamp += 33

                results = landmarker.detect_for_video(
                    mp_image,
                    frame_timestamp
                )

                angle = None
                feedback = "No person detected"

                if results.pose_landmarks:

                    landmarks = results.pose_landmarks[0]

                    # DRAW SKELETON

                    for start, end in CONNECTIONS:

                        p1 = landmarks[start]
                        p2 = landmarks[end]

                        if (
                            p1.visibility > 0.4
                            and p2.visibility > 0.4
                        ):

                            x1 = int(p1.x * width)
                            y1 = int(p1.y * height)

                            x2 = int(p2.x * width)
                            y2 = int(p2.y * height)

                            cv2.line(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                3
                            )

                    # DRAW LANDMARKS

                    for landmark in landmarks:

                        if landmark.visibility > 0.4:

                            x = int(landmark.x * width)
                            y = int(landmark.y * height)

                            cv2.circle(
                                frame,
                                (x, y),
                                6,
                                (0, 0, 255),
                                -1
                            )

                    shoulder = landmarks[shoulder_id]
                    elbow = landmarks[elbow_id]
                    wrist = landmarks[wrist_id]

                    visibility = min(
                        shoulder.visibility,
                        elbow.visibility,
                        wrist.visibility
                    )

                    if visibility > 0.5:

                        points = [

                            (shoulder.x, shoulder.y),
                            (elbow.x, elbow.y),
                            (wrist.x, wrist.y)

                        ]

                        angle = calculate_angle(
                            points[0],
                            points[1],
                            points[2]
                        )

                        old_reps = counter.reps

                        reps, feedback = counter.update(angle)
                        session.update_reps(counter.reps)

                        # REP ANNOUNCEMENT

                        if counter.reps > old_reps:

                            voice.speak(
                                f"Good repetition. "
                                f"{counter.reps} repetitions.",
                                force=True
                            )

                        # MOVEMENT FEEDBACK

                        if counter.stage != previous_stage:

                            if counter.stage == "EXTENDED":

                                voice.speak(
                                    "Now curl your arm.",
                                    force=True
                                )

                            elif counter.stage == "CURLED":

                                voice.speak(
                                    "Good. Extend your arm again.",
                                    force=True
                                )

                            previous_stage = counter.stage

                        elbow_x = int(elbow.x * width)
                        elbow_y = int(elbow.y * height)

                        cv2.putText(
                            frame,
                            f"Elbow: {int(angle)} deg",
                            (elbow_x + 20, elbow_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 0),
                            2
                        )

                    else:

                        feedback = "Keep your selected arm visible"

                # =================================================
                # DASHBOARD
                # =================================================

                elapsed = int(time.time() - start_time)

                dashboard = np.zeros(
                    (150, width, 3),
                    dtype=np.uint8
                )

                dashboard[:] = (25, 25, 25)

                cv2.putText(
                    dashboard,
                    "K-FITNESS | BICEP CURLS",
                    (25, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    3
                )

                cv2.putText(
                    dashboard,
                    f"REPS: {counter.reps}",
                    (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    dashboard,
                    f"ARM: {arm_name}",
                    (330, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    dashboard,
                    f"TIME: {elapsed}s",
                    (600, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    dashboard,
                    feedback,
                    (850, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                output = np.vstack(
                    (dashboard, frame)
                )

                cv2.imshow(
                    WINDOW_NAME,
                    output
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except Exception as error:

        print("\nExercise error:", error)

    finally:

        try:
            session.update_reps(counter.reps)
            saved_session_id = session.finish()
            print(f"[SESSION] Workout saved successfully! Session ID: {saved_session_id}")
            print("[SESSION] Exercise: Bicep Curls")
            print(f"[SESSION] Repetitions: {counter.reps}")
            print(f"[SESSION] Duration: {int(time.time() - start_time)} seconds")
        except Exception as save_error:
            print("[SESSION ERROR] Could not save workout:", repr(save_error))

        cap.release()
        cv2.destroyAllWindows()
        voice.close()

    print("\nBicep Curl Session Completed")
    print(f"Total repetitions: {counter.reps}")
    print(
        f"Duration: {int(time.time() - start_time)} seconds"
    )


if __name__ == "__main__":
    main()

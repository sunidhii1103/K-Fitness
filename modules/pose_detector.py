
import cv2
import mediapipe as mp
import time
import traceback
import numpy as np
import sys
from pathlib import Path

# ==========================================
# K-FITNESS | AI MOVEMENT COACH
# Pose Detection + Joint Angles + Squat Counter
# ==========================================

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"

# Import repetition counter
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from rep_counter import SquatRepCounter


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Pose model not found: {MODEL_PATH}"
    )


# ==========================================
# MEDIAPIPE CONFIGURATION
# ==========================================

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# SKELETON CONNECTIONS
# ==========================================

CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (27, 29), (29, 31),
    (24, 26), (26, 28),
    (28, 30), (30, 32),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22)
]


# ==========================================
# LANDMARK VISIBILITY
# ==========================================

def get_visibility(point):

    value = getattr(point, "visibility", 1.0)

    return 0.0 if value is None else value


def get_point(landmarks, index, width, height):

    point = landmarks[index]

    if get_visibility(point) < 0.5:
        return None

    # Convert normalized coordinates into pixels.
    # This avoids aspect-ratio distortion in angle calculation.
    return (
        point.x * width,
        point.y * height
    )


# ==========================================
# JOINT ANGLE CALCULATION
# ==========================================

def calculate_angle(a, b, c):

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return None

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))

    return round(float(angle), 1)


def get_joint_angle(
    landmarks,
    a_index,
    b_index,
    c_index,
    width,
    height
):

    a = get_point(landmarks, a_index, width, height)
    b = get_point(landmarks, b_index, width, height)
    c = get_point(landmarks, c_index, width, height)

    if a is None or b is None or c is None:
        return None

    return calculate_angle(a, b, c)


def calculate_all_angles(landmarks, width, height):

    return {

        "Left Knee": get_joint_angle(
            landmarks, 23, 25, 27, width, height
        ),

        "Right Knee": get_joint_angle(
            landmarks, 24, 26, 28, width, height
        ),

        "Left Hip": get_joint_angle(
            landmarks, 11, 23, 25, width, height
        ),

        "Right Hip": get_joint_angle(
            landmarks, 12, 24, 26, width, height
        ),

        "Left Elbow": get_joint_angle(
            landmarks, 11, 13, 15, width, height
        ),

        "Right Elbow": get_joint_angle(
            landmarks, 12, 14, 16, width, height
        ),

        "Left Shoulder": get_joint_angle(
            landmarks, 13, 11, 23, width, height
        ),

        "Right Shoulder": get_joint_angle(
            landmarks, 14, 12, 24, width, height
        )
    }


# ==========================================
# SELECT MOST RELIABLE KNEE ANGLE
# ==========================================

def select_knee_angle(landmarks, angles):

    left_visibility = min(
        get_visibility(landmarks[23]),
        get_visibility(landmarks[25]),
        get_visibility(landmarks[27])
    )

    right_visibility = min(
        get_visibility(landmarks[24]),
        get_visibility(landmarks[26]),
        get_visibility(landmarks[28])
    )

    candidates = []

    if angles["Left Knee"] is not None:
        candidates.append(
            (left_visibility, angles["Left Knee"])
        )

    if angles["Right Knee"] is not None:
        candidates.append(
            (right_visibility, angles["Right Knee"])
        )

    if not candidates:
        return None

    # Use the side with the clearest visible landmarks.
    return max(candidates, key=lambda item: item[0])[1]


# ==========================================
# DRAW SKELETON
# ==========================================

def draw_pose(frame, landmarks):

    height, width = frame.shape[:2]

    for start_idx, end_idx in CONNECTIONS:

        p1 = landmarks[start_idx]
        p2 = landmarks[end_idx]

        if (
            get_visibility(p1) < 0.5
            or get_visibility(p2) < 0.5
        ):
            continue

        x1 = int(p1.x * width)
        y1 = int(p1.y * height)

        x2 = int(p2.x * width)
        y2 = int(p2.y * height)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

    for point in landmarks:

        if get_visibility(point) < 0.5:
            continue

        x = int(point.x * width)
        y = int(point.y * height)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1
        )

    return frame


# ==========================================
# BODY VISIBILITY STATUS
# ==========================================

def check_body_visibility(landmarks):

    groups = {
        "Shoulders": [11, 12],
        "Hips": [23, 24],
        "Knees": [25, 26],
        "Ankles": [27, 28]
    }

    status = {}

    for name, indices in groups.items():

        status[name] = all(
            get_visibility(landmarks[i]) >= 0.5
            for i in indices
        )

    return status


# ==========================================
# DASHBOARD
# ==========================================

def draw_dashboard(
    frame,
    fps,
    angles,
    body_status,
    squat_result,
    people_count
):

    height, width = frame.shape[:2]

    # Header
    cv2.rectangle(
        frame,
        (0, 0),
        (width, 48),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        frame,
        "K-FITNESS | AI MOVEMENT COACH",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Camera information
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (15, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"People detected: {people_count}",
        (15, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    # Body visibility
    y = 145

    for name, visible in body_status.items():

        color = (
            (0, 255, 0)
            if visible
            else (0, 0, 255)
        )

        text = (
            f"{name}: OK"
            if visible
            else f"{name}: NOT VISIBLE"
        )

        cv2.putText(
            frame,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )

        y += 25

    # Squat information panel
    panel_x = width - 235

    cv2.rectangle(
        frame,
        (panel_x - 10, 55),
        (width - 5, 200),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "SQUAT MONITOR",
        (panel_x, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"REPS: {squat_result['reps']}",
        (panel_x, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"PHASE: {squat_result['phase']}",
        (panel_x, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1
    )

    angle = squat_result["angle"]

    angle_text = (
        f"Knee: {angle:.1f} deg"
        if angle is not None
        else "Knee: N/A"
    )

    cv2.putText(
        frame,
        angle_text,
        (panel_x, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 255),
        1
    )

    # Joint angles panel
    cv2.rectangle(
        frame,
        (panel_x - 10, 215),
        (width - 5, 445),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        "JOINT ANGLES",
        (panel_x, 243),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    y = 275

    for name, value in angles.items():

        text = (
            f"{name}: {value:.1f}"
            if value is not None
            else f"{name}: N/A"
        )

        cv2.putText(
            frame,
            text,
            (panel_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 255, 255),
            1
        )

        y += 21

    # Feedback
    cv2.putText(
        frame,
        squat_result["feedback"][:65],
        (15, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (0, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "Q / ESC: EXIT",
        (15, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

    return frame


# ==========================================
# MAIN APPLICATION
# ==========================================

def main():

    print("\n===================================")
    print("       K-FITNESS AI COACH")
    print("===================================")

    # Initialize squat counter
    squat_counter = SquatRepCounter()

    # Initialize webcam
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        raise RuntimeError(
            "Cannot access webcam. Check camera permissions."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window_name = "K-Fitness - AI Movement Coach"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        960,
        720
    )

    frame_count = 0
    fps = 0.0

    fps_start = time.perf_counter()
    session_start = time.perf_counter()
    last_timestamp = -1

    try:

        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            print("Camera initialized.")
            print("Pose detection enabled.")
            print("Joint angle calculation enabled.")
            print("Squat repetition counter enabled.")
            print("Press Q or ESC to exit.")

            while True:

                success, frame = camera.read()

                if not success:
                    print("Camera frame unavailable.")
                    break

                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                # Increasing video timestamp
                timestamp = int(
                    (time.perf_counter() - session_start) * 1000
                )

                timestamp = max(
                    timestamp,
                    last_timestamp + 1
                )

                last_timestamp = timestamp

                # Pose inference
                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp
                )

                people_count = len(result.pose_landmarks)

                # Default values
                angles = {
                    "Left Knee": None,
                    "Right Knee": None,
                    "Left Hip": None,
                    "Right Hip": None,
                    "Left Elbow": None,
                    "Right Elbow": None,
                    "Left Shoulder": None,
                    "Right Shoulder": None
                }

                body_status = {
                    "Shoulders": False,
                    "Hips": False,
                    "Knees": False,
                    "Ankles": False
                }

                if people_count > 0:

                    landmarks = result.pose_landmarks[0]

                    # Draw skeleton
                    frame = draw_pose(
                        frame,
                        landmarks
                    )

                    # Calculate angles
                    angles = calculate_all_angles(
                        landmarks,
                        width,
                        height
                    )

                    # Check body visibility
                    body_status = check_body_visibility(
                        landmarks
                    )

                    # Choose the clearest available knee
                    selected_knee = select_knee_angle(
                        landmarks,
                        angles
                    )

                    # Update repetition counter
                    squat_result = squat_counter.update(
                        selected_knee
                    )

                else:

                    squat_result = squat_counter.update(None)

                # FPS calculation
                frame_count += 1

                elapsed = time.perf_counter() - fps_start

                if elapsed >= 1.0:

                    fps = frame_count / elapsed

                    frame_count = 0
                    fps_start = time.perf_counter()

                # Draw dashboard
                frame = draw_dashboard(
                    frame,
                    fps,
                    angles,
                    body_status,
                    squat_result,
                    people_count
                )

                # Enlarge display without changing inference
                display_frame = cv2.resize(
                    frame,
                    (960, 720),
                    interpolation=cv2.INTER_LINEAR
                )

                cv2.imshow(
                    window_name,
                    display_frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    print("Exit requested.")
                    break

    except Exception:

        print("\n========== K-FITNESS ERROR ==========")
        traceback.print_exc()

    finally:

        camera.release()
        cv2.destroyAllWindows()

        print("\nCamera released.")
        print("K-Fitness stopped.")


if __name__ == "__main__":
    main()

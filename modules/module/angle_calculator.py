
import numpy as np

# ==========================================
# K-FITNESS - JOINT ANGLE CALCULATOR
# ==========================================

# MediaPipe Pose Landmark Indices
LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28
}


def calculate_angle(a, b, c):
    """
    Calculate angle at point B using three 2D points.
    Returns angle in degrees.
    """

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return None

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)

    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))

    return round(float(angle), 2)


def get_landmark_point(landmarks, name):
    """
    Extract normalized x,y coordinates
    from MediaPipe landmarks.
    """

    index = LANDMARKS[name]
    point = landmarks[index]

    visibility = getattr(point, "visibility", 1.0)

    if visibility is None or visibility < 0.5:
        return None

    return (point.x, point.y)


def get_joint_angle(landmarks, point_a, point_b, point_c):
    """
    Calculate an angle using named landmarks.
    """

    a = get_landmark_point(landmarks, point_a)
    b = get_landmark_point(landmarks, point_b)
    c = get_landmark_point(landmarks, point_c)

    if a is None or b is None or c is None:
        return None

    return calculate_angle(a, b, c)


def calculate_all_angles(landmarks):
    """
    Calculate all major joint angles.
    """

    angles = {

        # Knee angles
        "left_knee": get_joint_angle(
            landmarks,
            "left_hip",
            "left_knee",
            "left_ankle"
        ),

        "right_knee": get_joint_angle(
            landmarks,
            "right_hip",
            "right_knee",
            "right_ankle"
        ),

        # Hip angles
        "left_hip": get_joint_angle(
            landmarks,
            "left_shoulder",
            "left_hip",
            "left_knee"
        ),

        "right_hip": get_joint_angle(
            landmarks,
            "right_shoulder",
            "right_hip",
            "right_knee"
        ),

        # Elbow angles
        "left_elbow": get_joint_angle(
            landmarks,
            "left_shoulder",
            "left_elbow",
            "left_wrist"
        ),

        "right_elbow": get_joint_angle(
            landmarks,
            "right_shoulder",
            "right_elbow",
            "right_wrist"
        ),

        # Shoulder angles
        "left_shoulder": get_joint_angle(
            landmarks,
            "left_elbow",
            "left_shoulder",
            "left_hip"
        ),

        "right_shoulder": get_joint_angle(
            landmarks,
            "right_elbow",
            "right_shoulder",
            "right_hip"
        )
    }

    return angles


if __name__ == "__main__":

    # Simple mathematical test
    a = (0, 0)
    b = (0, 1)
    c = (1, 1)

    result = calculate_angle(a, b, c)

    print("K-Fitness Angle Calculator Test")
    print("Expected angle: 90 degrees")
    print("Calculated angle:", result)


"""
K-Fitness - Squat Form Analyzer

Analyzes:
1. Knee angle and squat depth
2. Forward torso lean
3. Left-right movement symmetry
4. Visibility of hip, knee and ankle
5. Overall heuristic form score

Input: MediaPipe PoseLandmarker landmarks.
Output: Score, feedback, detected issues and metrics.

This is a coaching aid, not a medical assessment.
"""

import math


# MediaPipe landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28

DEFAULT_VISIBILITY = 0.45


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_visibility(landmark):
    return float(
        getattr(landmark, "visibility", 0.0) or 0.0
    )


def get_point(landmark, width, height):
    return (
        float(landmark.x) * width,
        float(landmark.y) * height
    )


def calculate_angle(a, b, c):

    bax = a[0] - b[0]
    bay = a[1] - b[1]

    bcx = c[0] - b[0]
    bcy = c[1] - b[1]

    denominator = (
        math.hypot(bax, bay)
        * math.hypot(bcx, bcy)
    )

    if denominator < 1e-8:
        return None

    cosine = (
        bax * bcx + bay * bcy
    ) / denominator

    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def get_knee_angle(
    landmarks,
    hip_index,
    knee_index,
    ankle_index,
    width,
    height
):

    ids = (
        hip_index,
        knee_index,
        ankle_index
    )

    for index in ids:
        if get_visibility(landmarks[index]) < DEFAULT_VISIBILITY:
            return None

    hip = get_point(
        landmarks[hip_index],
        width,
        height
    )

    knee = get_point(
        landmarks[knee_index],
        width,
        height
    )

    ankle = get_point(
        landmarks[ankle_index],
        width,
        height
    )

    return calculate_angle(
        hip,
        knee,
        ankle
    )


# ============================================================
# MAIN SQUAT FORM ANALYZER
# ============================================================

def analyze_squat_form(
    landmarks,
    image_width,
    image_height
):

    if landmarks is None or len(landmarks) < 33:

        return {
            "score": None,
            "status": "INSUFFICIENT_DATA",
            "feedback": (
                "Move back so your hips, knees "
                "and ankles are visible."
            ),
            "issues": [
                "Insufficient pose landmarks"
            ],
            "metrics": {}
        }

    width = int(image_width)
    height = int(image_height)

    if width <= 0 or height <= 0:
        raise ValueError(
            "Image width and height must be positive."
        )

    # --------------------------------------------------------
    # 1. CALCULATE LEFT AND RIGHT KNEE ANGLES
    # --------------------------------------------------------

    left_angle = get_knee_angle(
        landmarks,
        LEFT_HIP,
        LEFT_KNEE,
        LEFT_ANKLE,
        width,
        height
    )

    right_angle = get_knee_angle(
        landmarks,
        RIGHT_HIP,
        RIGHT_KNEE,
        RIGHT_ANKLE,
        width,
        height
    )

    available_angles = []

    if left_angle is not None:
        available_angles.append(
            (left_angle, "LEFT")
        )

    if right_angle is not None:
        available_angles.append(
            (right_angle, "RIGHT")
        )

    if not available_angles:

        return {
            "score": None,
            "status": "INSUFFICIENT_DATA",
            "feedback": (
                "Keep your hip, knee and ankle visible. "
                "Move slightly farther from the camera."
            ),
            "issues": [
                "Knee angle unavailable"
            ],
            "metrics": {
                "left_knee_angle": left_angle,
                "right_knee_angle": right_angle
            }
        }

    # Select the side with the best landmark visibility.

    def side_visibility(side):

        if side == "LEFT":
            ids = (
                LEFT_HIP,
                LEFT_KNEE,
                LEFT_ANKLE
            )
        else:
            ids = (
                RIGHT_HIP,
                RIGHT_KNEE,
                RIGHT_ANKLE
            )

        return min(
            get_visibility(landmarks[i])
            for i in ids
        )

    chosen_angle, chosen_side = max(
        available_angles,
        key=lambda item: side_visibility(item[1])
    )

    # Select corresponding landmarks.

    if chosen_side == "LEFT":

        shoulder_i = LEFT_SHOULDER
        hip_i = LEFT_HIP
        knee_i = LEFT_KNEE
        ankle_i = LEFT_ANKLE

    else:

        shoulder_i = RIGHT_SHOULDER
        hip_i = RIGHT_HIP
        knee_i = RIGHT_KNEE
        ankle_i = RIGHT_ANKLE

    issues = []
    tips = []

    score = 100

    # --------------------------------------------------------
    # 2. CHECK LANDMARK VISIBILITY
    # --------------------------------------------------------

    shoulder_visible = (
        get_visibility(landmarks[shoulder_i])
        >= DEFAULT_VISIBILITY
    )

    hip_visible = (
        get_visibility(landmarks[hip_i])
        >= DEFAULT_VISIBILITY
    )

    knee_visible = (
        get_visibility(landmarks[knee_i])
        >= DEFAULT_VISIBILITY
    )

    ankle_visible = (
        get_visibility(landmarks[ankle_i])
        >= DEFAULT_VISIBILITY
    )

    # --------------------------------------------------------
    # 3. TORSO LEAN ANALYSIS
    # --------------------------------------------------------

    torso_lean = None

    if shoulder_visible and hip_visible:

        shoulder = get_point(
            landmarks[shoulder_i],
            width,
            height
        )

        hip = get_point(
            landmarks[hip_i],
            width,
            height
        )

        dx = shoulder[0] - hip[0]
        dy = hip[1] - shoulder[1]

        torso_lean = abs(
            math.degrees(
                math.atan2(
                    dx,
                    max(abs(dy), 1e-6)
                )
            )
        )

        if torso_lean > 55:

            issues.append(
                "Large forward torso lean"
            )

            tips.append(
                "Keep your chest lifted and reduce forward lean."
            )

            score -= 20

        elif torso_lean > 40:

            issues.append(
                "Some forward torso lean"
            )

            tips.append(
                "Try to keep your chest lifted as you squat."
            )

            score -= 10

    # --------------------------------------------------------
    # 4. SQUAT DEPTH ANALYSIS
    # --------------------------------------------------------

    if chosen_angle > 140:

        issues.append(
            "Not yet at squat depth"
        )

        tips.append(
            "Bend your knees a little more, "
            "only as far as comfortable."
        )

        score -= 15

    elif chosen_angle < 65:

        issues.append(
            "Very deep knee bend"
        )

        tips.append(
            "Use a comfortable range and avoid forcing extra depth."
        )

        score -= 5

    # --------------------------------------------------------
    # 5. LEFT-RIGHT SYMMETRY
    # --------------------------------------------------------

    symmetry_difference = None

    if (
        left_angle is not None
        and right_angle is not None
    ):

        symmetry_difference = abs(
            left_angle - right_angle
        )

        if symmetry_difference > 25:

            issues.append(
                "Noticeable left-right knee angle difference"
            )

            tips.append(
                "Distribute your movement more evenly between both legs."
            )

            score -= 15

        elif symmetry_difference > 15:

            issues.append(
                "Small left-right knee angle difference"
            )

            tips.append(
                "Check that both legs are moving evenly."
            )

            score -= 7

    # --------------------------------------------------------
    # 6. LOWER BODY VISIBILITY
    # --------------------------------------------------------

    if not (
        hip_visible
        and knee_visible
        and ankle_visible
    ):

        issues.append(
            "Lower-body landmarks partly occluded"
        )

        tips.append(
            "Adjust the camera so your hip, knee "
            "and ankle stay visible."
        )

        score -= 15

    # --------------------------------------------------------
    # 7. FINAL SCORE
    # --------------------------------------------------------

    score = max(
        0,
        min(100, score)
    )

    if not tips:

        feedback = (
            "Good visible alignment. "
            "Keep the movement controlled."
        )

        status = "GOOD"

    else:

        feedback = tips[0]

        status = "NEEDS_ATTENTION"

    # --------------------------------------------------------
    # 8. RETURN RESULTS
    # --------------------------------------------------------

    return {

        "score": score,

        "status": status,

        "feedback": feedback,

        "issues": issues,

        "metrics": {

            "selected_side": chosen_side,

            "knee_angle": round(
                chosen_angle, 1
            ),

            "left_knee_angle": (
                round(left_angle, 1)
                if left_angle is not None
                else None
            ),

            "right_knee_angle": (
                round(right_angle, 1)
                if right_angle is not None
                else None
            ),

            "torso_lean_degrees": (
                round(torso_lean, 1)
                if torso_lean is not None
                else None
            ),

            "left_right_angle_difference": (
                round(symmetry_difference, 1)
                if symmetry_difference is not None
                else None
            )
        },

        "limitations": [

            "Heuristic feedback only; not a medical assessment.",

            "Knee-in/knee-out alignment requires a clear front-facing camera view.",

            "Depth and torso-lean estimates depend on camera placement and landmark visibility."

        ]
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "K-Fitness Form Analyzer loaded successfully."
    )

    print(
        "Import analyze_squat_form() "
        "from your pose detector to use it."
    )

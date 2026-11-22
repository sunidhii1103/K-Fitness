
import math


class KneeExtensionAnalyzer:
    """
    Rule-based movement analysis for seated knee extensions.
    Uses MediaPipe pose landmarks.
    """

    VISIBILITY_THRESHOLD = 0.40

    def calculate_angle(self, a, b, c):
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot = ba[0] * bc[0] + ba[1] * bc[1]

        mag_a = math.hypot(*ba)
        mag_b = math.hypot(*bc)

        if mag_a == 0 or mag_b == 0:
            return None

        cosine = max(-1, min(1, dot / (mag_a * mag_b)))

        return math.degrees(math.acos(cosine))

    def analyze(self, landmarks, width, height):

        # Left shoulder, hip, knee, ankle
        ids = {
            "shoulder": 11,
            "hip": 23,
            "knee": 25,
            "ankle": 27
        }

        points = {}

        for name, index in ids.items():

            landmark = landmarks[index]

            visibility = getattr(landmark, "visibility", 0)

            if visibility < self.VISIBILITY_THRESHOLD:
                return self.result(
                    None,
                    "Move into camera view",
                    ["Body landmarks are not clearly visible"],
                    "INSUFFICIENT_DATA"
                )

            points[name] = (
                landmark.x * width,
                landmark.y * height
            )

        knee_angle = self.calculate_angle(
            points["hip"],
            points["knee"],
            points["ankle"]
        )

        torso_angle = self.calculate_angle(
            points["shoulder"],
            points["hip"],
            points["knee"]
        )

        issues = []

        # Approximate seated posture check
        if torso_angle is not None and torso_angle < 65:
            issues.append("Sit upright and avoid leaning too far forward")

        if knee_angle is not None:
            if knee_angle < 45:
                issues.append("Check your sitting position and leg visibility")

        if issues:
            feedback = issues[0]
            status = "NEEDS_ATTENTION"
            score = max(50, 100 - 20 * len(issues))
        else:
            feedback = "Posture looks stable"
            status = "GOOD"
            score = 100

        return self.result(
            score,
            feedback,
            issues,
            status,
            knee_angle,
            torso_angle
        )

    def result(
        self,
        score,
        feedback,
        issues,
        status,
        knee_angle=None,
        torso_angle=None
    ):
        return {
            "score": score,
            "feedback": feedback,
            "issues": issues,
            "status": status,
            "knee_angle": round(knee_angle, 1)
            if knee_angle is not None else None,
            "torso_angle": round(torso_angle, 1)
            if torso_angle is not None else None
        }

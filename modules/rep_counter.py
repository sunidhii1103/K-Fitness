
import time


class SquatRepCounter:
    """
    K-Fitness squat repetition counter.

    Counts one repetition after:
    Standing -> Descending -> Bottom -> Ascending -> Standing
    """

    def __init__(
        self,
        standing_threshold=155,
        bottom_threshold=110,
        min_rep_interval=0.8
    ):
        self.standing_threshold = standing_threshold
        self.bottom_threshold = bottom_threshold
        self.min_rep_interval = min_rep_interval

        self.reps = 0
        self.phase = "WAITING"

        self.smoothed_angle = None
        self.alpha = 0.35

        self.bottom_reached = False
        self.last_rep_time = 0

    def update(self, knee_angle):

        if knee_angle is None:
            return {
                "reps": self.reps,
                "phase": "NO KNEE DETECTED",
                "angle": None,
                "feedback": "Move fully into camera view"
            }

        # Smooth noisy angle measurements
        if self.smoothed_angle is None:
            self.smoothed_angle = knee_angle
        else:
            self.smoothed_angle = (
                self.alpha * knee_angle
                + (1 - self.alpha) * self.smoothed_angle
            )

        angle = self.smoothed_angle
        feedback = "Maintain controlled movement"

        # Initial standing position
        if self.phase == "WAITING":

            if angle >= self.standing_threshold:
                self.phase = "STANDING"
                feedback = "Ready! Start your squat"
            else:
                feedback = "Stand upright to begin"

        # Detect downward movement
        elif self.phase == "STANDING":

            if angle < self.standing_threshold:
                self.phase = "DESCENDING"
                feedback = "Lower yourself slowly"

        # Detect bottom position
        elif self.phase == "DESCENDING":

            if angle <= self.bottom_threshold:
                self.phase = "BOTTOM"
                self.bottom_reached = True
                feedback = "Bottom position detected"

            elif angle >= self.standing_threshold:
                # Movement reversed without reaching squat depth
                self.phase = "STANDING"
                feedback = "Try a controlled squat"

        # Detect upward movement
        elif self.phase == "BOTTOM":

            if angle > self.bottom_threshold:
                self.phase = "ASCENDING"
                feedback = "Push back up"

        # Complete repetition only after returning upright
        elif self.phase == "ASCENDING":

            if angle >= self.standing_threshold:

                current_time = time.monotonic()

                if (
                    self.bottom_reached
                    and current_time - self.last_rep_time
                    >= self.min_rep_interval
                ):
                    self.reps += 1
                    self.last_rep_time = current_time
                    feedback = "Great! Repetition completed"

                self.phase = "STANDING"
                self.bottom_reached = False

        return {
            "reps": self.reps,
            "phase": self.phase,
            "angle": round(angle, 1),
            "feedback": feedback
        }

    def reset(self):
        self.reps = 0
        self.phase = "WAITING"
        self.smoothed_angle = None
        self.bottom_reached = False
        self.last_rep_time = 0

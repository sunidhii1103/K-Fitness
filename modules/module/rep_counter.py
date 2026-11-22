
import time


class SquatRepCounter:

    def __init__(
        self,
        standing_threshold=155,
        bottom_threshold=125,
        min_rep_interval=1.5
    ):
        self.standing_threshold = standing_threshold
        self.bottom_threshold = bottom_threshold
        self.min_rep_interval = min_rep_interval

        self.reps = 0
        self.phase = "WAITING"

        self.smoothed_angle = None
        self.alpha = 0.35

        self.standing_frames = 0
        self.bottom_frames = 0

        self.last_rep_time = 0

        self.required_standing_frames = 4
        self.required_bottom_frames = 3

    def update(self, knee_angle):

        if knee_angle is None:
            return self.get_result(
                None, "Keep your hip, knee and ankle visible"
            )

        # Smooth noisy angle measurements
        if self.smoothed_angle is None:
            self.smoothed_angle = knee_angle
        else:
            self.smoothed_angle = (
                self.alpha * knee_angle
                + (1 - self.alpha) * self.smoothed_angle
            )

        angle = self.smoothed_angle
        feedback = "Move with control"

        # 1. INITIAL STANDING CALIBRATION
        if self.phase == "WAITING":

            if angle >= self.standing_threshold:
                self.standing_frames += 1
            else:
                self.standing_frames = 0

            if self.standing_frames >= 8:
                self.phase = "STANDING"
                self.standing_frames = 0
                feedback = "Ready. Start your squat"
            else:
                feedback = "Stand upright to begin"

        # 2. STANDING
        elif self.phase == "STANDING":

            feedback = "Good. Prepare for your next squat"

            if angle < 145:
                self.phase = "DESCENDING"
                self.bottom_frames = 0
                feedback = "Lower yourself slowly"

        # 3. DESCENDING
        elif self.phase == "DESCENDING":

            feedback = "Continue lowering slowly"

            if angle <= self.bottom_threshold:
                self.bottom_frames += 1

                if self.bottom_frames >= self.required_bottom_frames:
                    self.phase = "BOTTOM"
                    feedback = "Good depth. Push upward"

            else:
                self.bottom_frames = 0

                # If the user returns upright without
                # reaching the required depth, do not count.
                if angle >= self.standing_threshold:
                    self.phase = "STANDING"
                    feedback = "Try lowering a little further"

        # 4. BOTTOM
        elif self.phase == "BOTTOM":

            feedback = "Push upward steadily"

            if angle >= 140:
                self.phase = "ASCENDING"
                self.standing_frames = 0
                feedback = "Stand up steadily"

        # 5. ASCENDING
        elif self.phase == "ASCENDING":

            feedback = "Keep standing up"

            if angle >= self.standing_threshold:
                self.standing_frames += 1

                if self.standing_frames >= self.required_standing_frames:

                    now = time.time()

                    if now - self.last_rep_time >= self.min_rep_interval:
                        self.reps += 1
                        self.last_rep_time = now

                    self.phase = "STANDING"
                    self.standing_frames = 0
                    self.bottom_frames = 0

                    feedback = f"Repetition {self.reps} completed"

            else:
                self.standing_frames = 0

        return self.get_result(angle, feedback)

    def get_result(self, angle, feedback):

        return {
            "reps": self.reps,
            "phase": self.phase,
            "angle": angle,
            "feedback": feedback
        }

    def reset(self):

        self.reps = 0
        self.phase = "WAITING"

        self.smoothed_angle = None

        self.standing_frames = 0
        self.bottom_frames = 0

        self.last_rep_time = 0

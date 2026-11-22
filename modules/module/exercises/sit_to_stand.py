
import time


class SitToStandCounter:

    # Initial camera-calibration thresholds
    SITTING_THRESHOLD = 115
    STANDING_THRESHOLD = 150

    MIN_HOLD_FRAMES = 4
    MIN_REP_INTERVAL = 1.2

    def __init__(self):

        self.reps = 0
        self.phase = "WAITING"

        self.smoothed_angle = None
        self.alpha = 0.35

        self.sitting_frames = 0
        self.standing_frames = 0

        self.last_rep_time = 0

    def update(self, raw_angle):

        if raw_angle is None:
            return self.get_result(
                None,
                "Keep your hip, knee and ankle visible"
            )

        if not 40 <= raw_angle <= 180:
            return self.get_result(
                self.smoothed_angle,
                "Adjust your camera position"
            )

        # Smooth angle to reduce detection noise
        if self.smoothed_angle is None:

            self.smoothed_angle = raw_angle

        else:

            self.smoothed_angle = (
                self.alpha * raw_angle
                + (1 - self.alpha) * self.smoothed_angle
            )

        angle = self.smoothed_angle

        feedback = "Move slowly and steadily"

        # ----------------------------------------
        # INITIAL SITTING POSITION
        # ----------------------------------------

        if self.phase == "WAITING":

            if angle <= self.SITTING_THRESHOLD:
                self.sitting_frames += 1
            else:
                self.sitting_frames = 0

            if self.sitting_frames >= self.MIN_HOLD_FRAMES:

                self.phase = "SITTING"
                self.sitting_frames = 0

                feedback = "Starting position detected. Stand up slowly."

            else:

                feedback = "Sit comfortably on the chair"

        # ----------------------------------------
        # SITTING TO STANDING
        # ----------------------------------------

        elif self.phase == "SITTING":

            if angle >= self.STANDING_THRESHOLD:

                self.standing_frames += 1
                feedback = "Good! Hold your standing position."

                if self.standing_frames >= self.MIN_HOLD_FRAMES:

                    self.phase = "STANDING"
                    self.standing_frames = 0

                    feedback = "Now slowly sit back down."

            else:

                self.standing_frames = 0
                feedback = "Push through your feet and stand up slowly"

        # ----------------------------------------
        # STANDING TO SITTING
        # ----------------------------------------

        elif self.phase == "STANDING":

            if angle <= self.SITTING_THRESHOLD:

                self.sitting_frames += 1
                feedback = "Control your movement as you sit"

                if self.sitting_frames >= self.MIN_HOLD_FRAMES:

                    now = time.monotonic()

                    if now - self.last_rep_time >= self.MIN_REP_INTERVAL:

                        self.reps += 1
                        self.last_rep_time = now

                    self.phase = "SITTING"
                    self.sitting_frames = 0

                    feedback = "Repetition completed!"

            else:

                self.sitting_frames = 0
                feedback = "Slowly return to the chair"

        return self.get_result(angle, feedback)

    def get_result(self, angle, feedback):

        return {
            "reps": self.reps,
            "phase": self.phase,
            "angle": round(angle, 1) if angle is not None else None,
            "feedback": feedback
        }

    def reset(self):
        self.__init__()

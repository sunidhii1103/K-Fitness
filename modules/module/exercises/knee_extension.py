
import time


class KneeExtensionCounter:
    """
    Detects seated knee extensions using the hip-knee-ankle angle.

    One repetition:
    1. Start with the knee bent.
    2. Extend the leg.
    3. Return to the bent position.
    """

    EXTENDED_ANGLE = 155
    BENT_ANGLE = 115

    MIN_VALID_ANGLE = 35
    MAX_VALID_ANGLE = 180

    MIN_HOLD_FRAMES = 3
    MIN_REP_INTERVAL = 0.8

    def __init__(self):
        self.reps = 0
        self.phase = "WAITING"
        self.extended_frames = 0
        self.bent_frames = 0
        self.last_rep_time = 0

        self.smoothed_angle = None
        self.alpha = 0.3

    def update(self, raw_angle):

        if raw_angle is None:
            return self.result(
                None,
                "Keep hip, knee and ankle visible"
            )

        if not (
            self.MIN_VALID_ANGLE
            <= raw_angle
            <= self.MAX_VALID_ANGLE
        ):
            return self.result(
                self.smoothed_angle,
                "Adjust your position"
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

        if self.phase == "WAITING":

            if angle <= self.BENT_ANGLE:
                self.bent_frames += 1
            else:
                self.bent_frames = 0

            if self.bent_frames >= self.MIN_HOLD_FRAMES:
                self.phase = "BENT"
                self.bent_frames = 0

            feedback = "Begin with your knee comfortably bent"

        elif self.phase == "BENT":

            if angle >= self.EXTENDED_ANGLE:
                self.extended_frames += 1
                feedback = "Hold your leg extended"

                if self.extended_frames >= self.MIN_HOLD_FRAMES:
                    self.phase = "EXTENDED"
                    self.extended_frames = 0
            else:
                self.extended_frames = 0
                feedback = "Slowly straighten your knee"

        elif self.phase == "EXTENDED":

            if angle <= self.BENT_ANGLE:
                self.bent_frames += 1
                feedback = "Slowly bend your knee"

                if self.bent_frames >= self.MIN_HOLD_FRAMES:
                    now = time.monotonic()

                    if now - self.last_rep_time >= self.MIN_REP_INTERVAL:
                        self.reps += 1
                        self.last_rep_time = now

                    self.phase = "BENT"
                    self.bent_frames = 0
                    feedback = "Repetition completed"

            else:
                self.bent_frames = 0
                feedback = "Return your leg slowly"

        return self.result(angle, feedback)

    def result(self, angle, feedback):
        return {
            "reps": self.reps,
            "phase": self.phase,
            "angle": round(angle, 1) if angle is not None else None,
            "feedback": feedback
        }

    def reset(self):
        self.__init__()

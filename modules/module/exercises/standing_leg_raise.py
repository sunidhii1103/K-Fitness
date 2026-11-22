
import time


class StandingLegRaiseCounter:

    # Initial software calibration thresholds
    START_ANGLE = 165
    LIFT_ANGLE = 135

    MIN_HOLD_FRAMES = 3
    MIN_REP_INTERVAL = 1.0

    def __init__(self):

        self.reps = 0
        self.phase = "WAITING"

        self.smoothed_angle = None
        self.alpha = 0.35

        self.start_frames = 0
        self.lift_frames = 0

        self.last_rep_time = 0

    def update(self, raw_angle):

        if raw_angle is None:
            return self.get_result(
                None,
                "Keep your hip and knee visible"
            )

        if not 20 <= raw_angle <= 180:
            return self.get_result(
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

        # Starting position: leg down
        if self.phase == "WAITING":

            if angle >= self.START_ANGLE:
                self.start_frames += 1
            else:
                self.start_frames = 0

            if self.start_frames >= self.MIN_HOLD_FRAMES:

                self.phase = "STANDING"
                self.start_frames = 0

                feedback = "Starting position detected"

            else:
                feedback = "Stand upright with your leg down"

        # Lift the leg forward
        elif self.phase == "STANDING":

            if angle <= self.LIFT_ANGLE:

                self.lift_frames += 1
                feedback = "Good lift. Hold briefly."

                if self.lift_frames >= self.MIN_HOLD_FRAMES:

                    self.phase = "LIFTED"
                    self.lift_frames = 0

            else:

                self.lift_frames = 0
                feedback = "Slowly lift your leg forward"

        # Return the leg
        elif self.phase == "LIFTED":

            if angle >= self.START_ANGLE:

                self.start_frames += 1
                feedback = "Return your leg slowly"

                if self.start_frames >= self.MIN_HOLD_FRAMES:

                    now = time.monotonic()

                    if now - self.last_rep_time >= self.MIN_REP_INTERVAL:
                        self.reps += 1
                        self.last_rep_time = now

                    self.phase = "STANDING"
                    self.start_frames = 0

                    feedback = "Repetition completed!"

            else:

                self.start_frames = 0
                feedback = "Continue lowering your leg"

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

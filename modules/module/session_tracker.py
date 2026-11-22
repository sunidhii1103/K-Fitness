
from datetime import datetime
from pathlib import Path
import sys

# Ensure database.py can be imported from this folder
MODULE_DIR = Path(__file__).resolve().parent

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import database


class WorkoutSessionTracker:

    def __init__(self, exercise_name):

        database.initialize_database()

        self.exercise_name = exercise_name
        self.started_at = datetime.now()

        self.reps = 0
        self.form_scores = []
        self.form_corrections = 0

        self.finished = False

        print(f"[SESSION] Started: {self.exercise_name}")

    def update_reps(self, reps):
        """Update the current repetition count."""
        self.reps = max(0, int(reps))

    def add_form_score(self, score):
        """Record a valid form score between 0 and 100."""
        if score is not None:
            try:
                score = float(score)

                if 0 <= score <= 100:
                    self.form_scores.append(score)

            except (TypeError, ValueError):
                pass

    def add_correction(self, count=1):
        """Record the number of form corrections."""
        self.form_corrections += max(0, int(count))

    def finish(self):
        """Save the completed workout and return its database ID."""

        if self.finished:
            return None

        self.finished = True

        ended_at = datetime.now()

        duration = (
            ended_at - self.started_at
        ).total_seconds()

        avg_score = None
        best_score = None
        lowest_score = None

        if self.form_scores:

            avg_score = round(
                sum(self.form_scores) / len(self.form_scores),
                2
            )

            best_score = int(max(self.form_scores))
            lowest_score = int(min(self.form_scores))

        try:

            session_id = database.save_session(
                started_at=self.started_at.isoformat(timespec="seconds"),
                ended_at=ended_at.isoformat(timespec="seconds"),
                duration_seconds=round(duration, 2),
                exercise=self.exercise_name,
                reps=self.reps,
                avg_form_score=avg_score,
                best_form_score=best_score,
                lowest_form_score=lowest_score,
                form_corrections=self.form_corrections
            )

            print("\n[SESSION] Workout saved successfully!")
            print(f"[SESSION] Exercise: {self.exercise_name}")
            print(f"[SESSION] Repetitions: {self.reps}")
            print(f"[SESSION] Duration: {round(duration, 2)} seconds")
            print(f"[SESSION] Database ID: {session_id}")

            return session_id

        except Exception as error:

            print(f"\n[SESSION] Database save failed: {error}")
            return None

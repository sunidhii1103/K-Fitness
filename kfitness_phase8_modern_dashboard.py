from pathlib import Path
import sqlite3
import sys
import threading
import time
import math
import json
import base64
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta

import cv2
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import streamlit.components.v1 as components
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# ============================================================
# PROJECT PATHS
# ============================================================
ROOT = Path(__file__).resolve().parent
MODULE_DIR = ROOT / "modules" / "module"
MODEL_PATH = ROOT / "pose_landmarker_lite.task"
PROFILE_DB = ROOT / "kfitness_profiles.db"
SESSION_DB = ROOT / "kfitness_sessions.db"
DEFAULT_IP_URL = "http://10.30.79.120:8080/video"

EXERCISES = {
    "Squats": {
        "a": (24, 26, 28), "start": 105, "finish": 155,
        "start_label": "Squat down", "finish_label": "Stand up",
        "description": "Bend and extend your knees with controlled movement."
    },
    "Seated Knee Extensions": {
        "a": (23, 25, 27), "start": 100, "finish": 120,
        "start_label": "Bend your knee", "finish_label": "Extend your leg",
        "description": "Sit sideways to the camera and bend/straighten one knee."
    },
    "Standing Leg Raises": {
        "a": (11, 23, 25), "start": 135, "finish": 165,
        "start_label": "Lift your leg", "finish_label": "Lower your leg",
        "description": "Stand sideways and lift one leg forward in a controlled way."
    },
    "Sit-to-Stand": {
        "a": (24, 26, 28), "start": 115, "finish": 150,
        "start_label": "Sit down", "finish_label": "Stand up",
        "description": "Use a stable chair and move between sitting and standing."
    },
    "Bicep Curls": {
        "a": (12, 14, 16), "start": 65, "finish": 150,
        "start_label": "Curl your arm", "finish_label": "Extend your arm",
        "description": "Keep your upper arm steady while bending and extending the elbow."
    },
}
GOALS = [
    "General fitness", "Weight management", "Strength building",
    "Flexibility and mobility", "Physiotherapy and rehabilitation"
]

EXPERIENCE = ["Beginner", "Intermediate", "Advanced"]

PLAN_LIBRARY = {
    "General fitness": [
        ("Squats", 2, 8, 60),
        ("Standing Leg Raises", 2, 8, 45),
        ("Bicep Curls", 2, 10, 45),
    ],
    "Weight management": [
        ("Sit-to-Stand", 2, 10, 60),
        ("Standing Leg Raises", 2, 10, 45),
        ("Squats", 2, 8, 60),
    ],
    "Strength building": [
        ("Squats", 3, 8, 75),
        ("Bicep Curls", 3, 10, 60),
        ("Sit-to-Stand", 2, 8, 60),
    ],
    "Flexibility and mobility": [
        ("Seated Knee Extensions", 2, 8, 45),
        ("Standing Leg Raises", 2, 8, 45),
        ("Sit-to-Stand", 2, 6, 60),
    ],
    "Physiotherapy and rehabilitation": [
        ("Seated Knee Extensions", 1, 8, 60),
        ("Standing Leg Raises", 1, 8, 60),
        ("Sit-to-Stand", 1, 6, 60),
    ],
}

OLLAMA_URL_DEFAULT = "http://localhost:11434"
OLLAMA_MODEL_DEFAULT = "qwen2.5vl:3b"
OLLAMA_INFERENCE_LOCK = threading.Lock()

# ============================================================
# DATABASE
# ============================================================
def profile_conn():
    c = sqlite3.connect(PROFILE_DB)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with profile_conn() as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_code TEXT UNIQUE,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT,
            height_cm REAL NOT NULL,
            weight_kg REAL NOT NULL,
            goal TEXT NOT NULL,
            experience TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        c.commit()

    SESSION_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SESSION_DB) as c:
        # Compatible with the existing modules/module/database.py schema.
        c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            duration_seconds REAL DEFAULT 0,
            exercise TEXT NOT NULL,
            reps INTEGER DEFAULT 0,
            avg_form_score REAL,
            best_form_score REAL,
            lowest_form_score REAL,
            form_corrections INTEGER DEFAULT 0
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS session_profiles (
            session_id INTEGER PRIMARY KEY,
            user_code TEXT NOT NULL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS workout_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_code TEXT NOT NULL,
            title TEXT NOT NULL,
            goal TEXT NOT NULL,
            created_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS workout_plan_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id INTEGER NOT NULL,
            exercise TEXT NOT NULL,
            sets INTEGER NOT NULL,
            reps_per_set INTEGER NOT NULL,
            rest_seconds INTEGER NOT NULL,
            sort_order INTEGER NOT NULL,
            FOREIGN KEY(plan_id) REFERENCES workout_plans(id) ON DELETE CASCADE
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS session_coaching (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            user_code TEXT NOT NULL,
            exercise TEXT NOT NULL,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            feedback TEXT NOT NULL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS rep_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            rep_number INTEGER NOT NULL,
            duration_seconds REAL,
            range_of_motion REAL,
            movement_score REAL,
            feedback TEXT,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )""")
        c.commit()


def profiles():
    with profile_conn() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM profiles ORDER BY name COLLATE NOCASE"
        ).fetchall()]


def save_profile(data, code=None):
    with profile_conn() as c:
        if code:
            c.execute("""
            UPDATE profiles SET name=?, age=?, gender=?, height_cm=?,
            weight_kg=?, goal=?, experience=? WHERE user_code=?
            """, (*data, code))
            c.commit()
            return code
        cur = c.execute("""
        INSERT INTO profiles
        (user_code,name,age,gender,height_cm,weight_kg,goal,experience,created_at)
        VALUES ('PENDING',?,?,?,?,?,?,?,?)
        """, (*data, datetime.now().isoformat(timespec="seconds")))
        new_code = f"KF{cur.lastrowid:05d}"
        c.execute("UPDATE profiles SET user_code=? WHERE id=?",
                  (new_code, cur.lastrowid))
        c.commit()
        return new_code


def save_workout(user_code, exercise, started, reps, rep_events=None):
    ended = datetime.now()
    duration = max(0, time.time() - started)
    rep_events = rep_events or []
    scores = [float(x["score"]) for x in rep_events if x.get("score") is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else None
    best_score = round(max(scores), 1) if scores else None
    lowest_score = round(min(scores), 1) if scores else None
    corrections = sum(1 for x in rep_events if x.get("feedback"))
    SESSION_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SESSION_DB) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            duration_seconds REAL DEFAULT 0,
            exercise TEXT NOT NULL,
            reps INTEGER DEFAULT 0,
            avg_form_score REAL,
            best_form_score REAL,
            lowest_form_score REAL,
            form_corrections INTEGER DEFAULT 0
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS session_profiles (
            session_id INTEGER PRIMARY KEY,
            user_code TEXT NOT NULL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS rep_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            rep_number INTEGER NOT NULL,
            duration_seconds REAL,
            range_of_motion REAL,
            movement_score REAL,
            feedback TEXT
        )""")
        cur = c.execute("""
        INSERT INTO sessions
        (started_at, ended_at, duration_seconds, exercise, reps,
         avg_form_score, best_form_score, lowest_form_score, form_corrections)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (datetime.fromtimestamp(started).isoformat(timespec="seconds"),
              ended.isoformat(timespec="seconds"), round(duration, 2),
              exercise, int(reps), avg_score, best_score, lowest_score, corrections))
        session_id = cur.lastrowid
        c.execute(
            "INSERT OR REPLACE INTO session_profiles(session_id,user_code) VALUES (?,?)",
            (session_id, user_code)
        )
        for item in rep_events:
            c.execute("""
                INSERT INTO rep_metrics
                (session_id,rep_number,duration_seconds,range_of_motion,movement_score,feedback)
                VALUES(?,?,?,?,?,?)
            """, (session_id, item["rep"], item["duration"], item["rom"],
                  item["score"], item.get("feedback", "")))
        c.commit()
    return session_id, duration


def get_user_sessions(user_code):
    if not SESSION_DB.exists():
        return []
    with sqlite3.connect(SESSION_DB) as c:
        c.row_factory = sqlite3.Row
        try:
            linked = [dict(r) for r in c.execute("""
                SELECT s.* FROM sessions s
                JOIN session_profiles p ON p.session_id=s.id
                WHERE p.user_code=?
                ORDER BY s.started_at DESC
            """, (user_code,)).fetchall()]
            if linked:
                return linked

            legacy = [dict(r) for r in c.execute("""
                SELECT s.* FROM sessions s
                LEFT JOIN session_profiles p ON p.session_id=s.id
                WHERE p.session_id IS NULL
                ORDER BY s.started_at DESC
            """).fetchall()]
            for row in legacy:
                row["_legacy_unlinked"] = True
            return legacy
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"Could not read workout history from {SESSION_DB}: {exc}"
            ) from exc


def get_rep_metrics(session_id):
    with sqlite3.connect(SESSION_DB) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(
            "SELECT * FROM rep_metrics WHERE session_id=? ORDER BY rep_number",
            (session_id,)
        ).fetchall()]


def get_all_rep_metrics(user_code):
    with sqlite3.connect(SESSION_DB) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute("""
            SELECT rm.*, s.exercise, s.started_at
            FROM rep_metrics rm
            JOIN sessions s ON s.id=rm.session_id
            JOIN session_profiles p ON p.session_id=s.id
            WHERE p.user_code=?
            ORDER BY s.started_at, rm.rep_number
        """, (user_code,)).fetchall()]


def save_plan(user_code, title, goal, items):
    with sqlite3.connect(SESSION_DB) as c:
        c.execute("UPDATE workout_plans SET active=0 WHERE user_code=?", (user_code,))
        cur = c.execute(
            "INSERT INTO workout_plans(user_code,title,goal,created_at,active) VALUES(?,?,?,?,1)",
            (user_code, title, goal, datetime.now().isoformat(timespec="seconds"))
        )
        plan_id = cur.lastrowid
        for order, item in enumerate(items):
            c.execute("""
                INSERT INTO workout_plan_items
                (plan_id,exercise,sets,reps_per_set,rest_seconds,sort_order)
                VALUES(?,?,?,?,?,?)
            """, (plan_id, item["exercise"], int(item["sets"]),
                  int(item["reps"]), int(item["rest"]), order))
        c.commit()
    return plan_id


def get_active_plan(user_code):
    with sqlite3.connect(SESSION_DB) as c:
        c.row_factory = sqlite3.Row
        plan = c.execute("""
            SELECT * FROM workout_plans
            WHERE user_code=? AND active=1 ORDER BY id DESC LIMIT 1
        """, (user_code,)).fetchone()
        if not plan:
            return None, []
        items = c.execute("""
            SELECT * FROM workout_plan_items
            WHERE plan_id=? ORDER BY sort_order
        """, (plan["id"],)).fetchall()
        return dict(plan), [dict(x) for x in items]


def save_coaching_note(user_code, exercise, source, feedback, session_id=None):
    with sqlite3.connect(SESSION_DB) as c:
        c.execute("""
            INSERT INTO session_coaching
            (session_id,user_code,exercise,created_at,source,feedback)
            VALUES(?,?,?,?,?,?)
        """, (session_id, user_code, exercise,
              datetime.now().isoformat(timespec="seconds"), source, feedback))
        c.commit()


def get_coaching_notes(user_code):
    with sqlite3.connect(SESSION_DB) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute("""
            SELECT * FROM session_coaching
            WHERE user_code=? ORDER BY id DESC LIMIT 30
        """, (user_code,)).fetchall()]


def recommend_plan(profile):
    """Transparent profile/goal-based starting template; user can edit every value."""
    base = PLAN_LIBRARY.get(profile.get("goal"), PLAN_LIBRARY["General fitness"])
    level = profile.get("experience", "Beginner")
    age = int(profile.get("age") or 30)

    if level == "Beginner" or age >= 60:
        scale = 1
    elif level == "Advanced":
        scale = 3
    else:
        scale = 2

    items = []
    for exercise, sets, reps, rest in base:
        items.append({
            "exercise": exercise,
            "sets": min(3, scale if level != "Beginner" else sets),
            "reps": reps if level != "Advanced" else min(12, reps + 2),
            "rest": max(rest, 60 if age >= 60 else rest),
        })
    return items



def generate_ai_workout_plan(profile, preferences,
                             base_url=OLLAMA_URL_DEFAULT,
                             model=OLLAMA_MODEL_DEFAULT):
    """Generate and validate a personalized workout plan with local Ollama."""
    allowed_exercises = list(EXERCISES.keys())

    prompt = f"""
You are K-Fitness, a cautious fitness-planning assistant.

Use ONLY these supported exercises:
{json.dumps(allowed_exercises)}

USER PROFILE:
- Age: {profile.get("age")}
- Gender: {profile.get("gender") or "Not specified"}
- Height: {profile.get("height_cm")} cm
- Weight: {profile.get("weight_kg")} kg
- Goal: {profile.get("goal")}
- Experience: {profile.get("experience")}

USER PREFERENCES:
- Activity level: {preferences.get("activity_level")}
- Workout days per week: {preferences.get("days_per_week")}
- Session duration: {preferences.get("session_minutes")} minutes
- Equipment: {preferences.get("equipment")}
- Physical limitations/restrictions: {preferences.get("limitations")}

RULES:
- Return 2 to 5 exercise entries.
- Each exercise must be from the supported list.
- Sets: integer 1 to 4; reps: integer 4 to 15; rest: integer 30 to 180 seconds.
- Keep the plan conservative and appropriate to the stated experience.
- Do not infer ability from gender, age, height or weight alone.
- Do not diagnose, prescribe treatment or claim to prevent injuries.
- If a user mentions pain, injury, a medical condition or rehabilitation restriction,
  clearly say a qualified healthcare professional should approve the plan.
- Return JSON only, with no Markdown or extra text.

Required JSON:
{{
  "title": "Personalized plan title",
  "rationale": "Brief explanation of the plan",
  "weekly_frequency": "Suggested weekly frequency",
  "items": [
    {{"exercise": "Squats", "sets": 2, "reps": 8, "rest": 60}}
  ],
  "safety_notes": "Safety reminder"
}}
"""

    payload = json.dumps({
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": 0.2}
    }).encode("utf-8")

    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with OLLAMA_INFERENCE_LOCK:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))

        content = result.get("message", {}).get("content", "").strip()
        if not content:
            raise ValueError("The AI returned an empty response.")

        if content.startswith("```"):
            content = content.replace("```json", "", 1)
            content = content.replace("```", "").strip()

        data = json.loads(content)
        items = data.get("items")

        if not isinstance(items, list) or not 2 <= len(items) <= 5:
            raise ValueError("AI must return between 2 and 5 exercises.")

        validated_items = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("An exercise entry was not valid JSON.")

            exercise = item.get("exercise")
            if exercise not in allowed_exercises:
                raise ValueError(f"Unsupported exercise returned: {exercise}")

            sets = int(item.get("sets"))
            reps = int(item.get("reps"))
            rest = int(item.get("rest"))

            if not 1 <= sets <= 4:
                raise ValueError("AI returned sets outside the allowed range.")
            if not 4 <= reps <= 15:
                raise ValueError("AI returned repetitions outside the allowed range.")
            if not 30 <= rest <= 180:
                raise ValueError("AI returned rest outside the allowed range.")

            validated_items.append({
                "exercise": exercise,
                "sets": sets,
                "reps": reps,
                "rest": rest
            })

        data["items"] = validated_items
        data["title"] = str(data.get("title") or f"{profile.get('goal')} plan")[:80]
        data["rationale"] = str(data.get("rationale") or "")
        data["weekly_frequency"] = str(data.get("weekly_frequency") or "")
        data["safety_notes"] = str(data.get("safety_notes") or "")
        return data

    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Start Ollama and check that "
            f"the model '{model}' is installed. Details: {exc}"
        ) from exc
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"AI returned an invalid workout plan: {exc}") from exc



def ollama_vision_feedback(frame_bgr, exercise, angle, reps, feedback,
                           base_url=OLLAMA_URL_DEFAULT, model=OLLAMA_MODEL_DEFAULT):
    """Optional local VLM call. The frame is sent only to the configured local Ollama server."""
    import cv2
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("Could not encode the camera frame.")
    image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    prompt = (
        f"You are a cautious exercise-form assistant, not a clinician. "
        f"Review this single frame for the exercise: {exercise}. "
        f"Current measured joint angle: {angle if angle is not None else 'unavailable'} degrees. "
        f"Repetition count: {reps}. Rule-based cue: {feedback}. "
        "Only comment on visible posture and camera framing. Do not infer pain, injury, "
        "diagnosis, age, or hidden body positions. If the view is insufficient, say so. "
        "Give at most 3 concise, gentle suggestions and remind the user to stop if uncomfortable."
    )
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
    }).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        # Serialize manual and automatic requests so the local GPU model is not
        # asked to run two vision generations at the same time.
        with OLLAMA_INFERENCE_LOCK:
            with urllib.request.urlopen(request, timeout=90) as response:
                result = json.loads(response.read().decode("utf-8"))
        return result.get("message", {}).get("content", "The VLM returned no feedback.")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach Ollama. Install/start Ollama and pull the configured vision model. "
            f"Details: {exc}"
        ) from exc


class AsyncVLMCoach:
    """Runs selected-frame VLM analysis away from the camera processing thread."""
    def __init__(self, user_code, exercise, base_url, model):
        self.user_code = user_code
        self.exercise = exercise
        self.base_url = base_url
        self.model = model
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kfitness-vlm")
        self.future = None
        self.lock = threading.Lock()
        self.latest_feedback = "Waiting for the next analysis checkpoint."
        self.status = "Ready"
        self.last_analyzed_rep = 0

    def _analyze_and_save(self, frame, angle, reps, cue):
        note = ollama_vision_feedback(
            frame, self.exercise, angle, reps, cue, self.base_url, self.model
        )
        save_coaching_note(
            self.user_code, self.exercise, f"Automatic VLM · rep {reps}", note
        )
        return note

    def submit(self, frame, angle, reps, cue):
        with self.lock:
            if self.future is not None and not self.future.done():
                self.status = "Analysis in progress"
                return False
            if self.future is not None and self.future.done():
                try:
                    self.latest_feedback = self.future.result()
                    self.status = "Latest analysis complete"
                except Exception as exc:
                    self.latest_feedback = f"AI analysis unavailable: {exc}"
                    self.status = "Analysis failed"
            self.future = self.executor.submit(
                self._analyze_and_save, frame.copy(), angle, reps, cue
            )
            self.status = f"Analyzing repetition {reps}"
            self.last_analyzed_rep = reps
            return True

    def poll(self):
        with self.lock:
            if self.future is not None and self.future.done():
                try:
                    self.latest_feedback = self.future.result()
                    self.status = "Latest analysis complete"
                except Exception as exc:
                    self.latest_feedback = f"AI analysis unavailable: {exc}"
                    self.status = "Analysis failed"
                self.future = None
            return self.status, self.latest_feedback

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)




# ============================================================
# BROWSER VOICE COACH + WORKOUT REVIEW
# Uses the browser's built-in speech synthesis; no extra Python
# TTS package is required. Speech is generated on the user's device.
# ============================================================
def speak_in_browser(text, event_key):
    """Speak one message in the browser, without queueing duplicates."""
    if not text:
        return
    message = str(text).strip()
    if not message:
        return
    message_json = json.dumps(message)
    key_json = json.dumps(str(event_key))
    html = f"""
    <script>
    (() => {{
      const message = {message_json};
      const eventKey = {key_json};
      const storageKey = "kfitness-spoken:" + eventKey;
      try {{
        if (window.localStorage.getItem(storageKey) === "1") return;
        if (!("speechSynthesis" in window)) return;
        window.localStorage.setItem(storageKey, "1");
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(message);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        window.speechSynthesis.speak(utterance);
      }} catch (error) {{
        console.warn("K-Fitness voice coach:", error);
      }}
    }})();
    </script>
    """
    components.html(html, height=0, scrolling=False)


def create_workout_review(exercise, reps, duration, rep_events):
    """Create a concise, evidence-based text review from recorded session data."""
    events = rep_events or []
    scores = [float(item["score"]) for item in events if item.get("score") is not None]
    corrections = [item.get("feedback", "").strip() for item in events
                   if item.get("feedback", "").strip()]
    avg_score = round(sum(scores) / len(scores), 1) if scores else None
    minutes, seconds = divmod(int(duration), 60)
    lines = [
        f"Workout review for {exercise}.",
        f"You completed {int(reps)} repetitions in {minutes} minutes and {seconds} seconds."
    ]
    if avg_score is not None:
        lines.append(f"Your experimental movement score averaged {avg_score} out of 100.")
    else:
        lines.append("A movement score was not available for this session.")
    if corrections:
        lines.append("Coaching observations: " + " ".join(dict.fromkeys(corrections)))
    else:
        lines.append("No rep-level correction was recorded. Keep using a comfortable, controlled pace.")
    lines.append("These are camera-based practice cues, not a medical assessment.")
    return {
        "exercise": exercise,
        "reps": int(reps),
        "duration": float(duration),
        "average_score": avg_score,
        "text": " ".join(lines),
    }


# ============================================================
# SHARED POSE / REP PROCESSOR
# ============================================================
SKELETON = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),(23,25),(25,27),
    (24,26),(26,28),(27,29),(29,31),(28,30),
    (30,32),(15,17),(15,19),(16,18),(16,20)
]


def angle_between(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a-b, c-b
    den = np.linalg.norm(ba) * np.linalg.norm(bc)
    if den < 1e-8:
        return None
    cosine = np.clip(np.dot(ba, bc) / den, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class PoseWorkout:
    def __init__(self, exercise, vlm_enabled=False, vlm_every_n_reps=3,
                 ollama_url=OLLAMA_URL_DEFAULT, ollama_model=OLLAMA_MODEL_DEFAULT,
                 user_code=None):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"MediaPipe model missing: {MODEL_PATH}")
        self.exercise = exercise
        self.config = EXERCISES[exercise]
        self.reps = 0
        self.phase = "WAITING"
        self.angle = None
        self.feedback = "Move into the camera frame"
        self.side = "--"
        self.last_rep = 0.0
        self.timestamp_ms = 0
        self.started = time.time()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.rule_cue = "Move into the camera frame"
        self.min_angle = None
        self.max_angle = None
        self.cycle_min_angle = None
        self.cycle_max_angle = None
        self.cycle_started_at = None
        self.rep_events = []
        self.vlm_enabled = bool(vlm_enabled and user_code)
        self.vlm_every_n_reps = max(1, int(vlm_every_n_reps))
        self.vlm_coach = (
            AsyncVLMCoach(user_code, exercise, ollama_url, ollama_model)
            if self.vlm_enabled else None
        )
        BaseOptions = mp.tasks.BaseOptions
        Options = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
            Options(
                base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    def process(self, frame):
        h, w = frame.shape[:2]
        # Smaller frames reduce MediaPipe inference time, especially on phone IP streams.
        if w > 640:
            frame = cv2.resize(frame, (640, int(h*640/w)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp_ms = max(self.timestamp_ms + 1, int(time.monotonic()*1000))
        result = self.landmarker.detect_for_video(image, self.timestamp_ms)
        reps_before_frame = self.reps

        with self.lock:
            if result.pose_landmarks:
                lm = result.pose_landmarks[0]
                for i, j in SKELETON:
                    if lm[i].visibility > 0.4 and lm[j].visibility > 0.4:
                        p1 = (int(lm[i].x*w), int(lm[i].y*h))
                        p2 = (int(lm[j].x*w), int(lm[j].y*h))
                        cv2.line(frame, p1, p2, (80, 220, 120), 2)
                ids = self.config["a"]
                candidates = []
                if self.exercise == "Standing Leg Raises":
                    side_triples = [
                        ("LEFT", (11, 23, 25)),
                        ("RIGHT", (12, 24, 26)),
                    ]
                else:
                    side_triples = [
                        ("LEFT", ids),
                        ("RIGHT", tuple(x + 1 if x % 2 else x - 1 for x in ids)),
                    ]

                for label, triple in side_triples:
                    try:
                        pts = [lm[k] for k in triple]
                        vis = min(p.visibility for p in pts)
                        if vis >= 0.4:
                            coords = [(p.x*w, p.y*h) for p in pts]
                            ang = angle_between(*coords)
                            if ang is not None:
                                candidates.append((vis, ang, label, coords))
                    except (IndexError, AttributeError):
                        pass
                if candidates:
                    _, ang, side, coords = max(candidates, key=lambda x:x[0])
                    self.side = side
                    self.angle = ang
                    self.min_angle = ang if self.min_angle is None else min(self.min_angle, ang)
                    self.max_angle = ang if self.max_angle is None else max(self.max_angle, ang)
                    self._update_rep(ang)
                    self.rule_cue = self.feedback
                    pts = [(int(x), int(y)) for x,y in coords]
                    cv2.line(frame, pts[0], pts[1], (255, 80, 220), 4)
                    cv2.line(frame, pts[1], pts[2], (255, 80, 220), 4)
                    for pt in pts:
                        cv2.circle(frame, pt, 7, (0, 255, 255), -1)
                else:
                    self.angle = None
                    self.feedback = "Keep the target joints visible"
            else:
                self.angle = None
                self.feedback = "Person not detected"

            elapsed = int(time.time() - self.started)
            angle_text = f"{self.angle:.1f} deg" if self.angle is not None else "--"
            lines = [
                f"K-FITNESS | {self.exercise}",
                f"Reps: {self.reps}     Phase: {self.phase}",
                f"Angle: {angle_text}     Side: {self.side}",
                self.feedback[:55],
                f"Duration: {elapsed//60:02d}:{elapsed%60:02d}",
            ]
            for n, line in enumerate(lines):
                cv2.putText(frame, line, (18, 30+n*31),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255,255,255), 2, cv2.LINE_AA)
        with self.lock:
            self.latest_frame = frame.copy()

        # Automatic VLM checkpoint: only every N completed reps, never on every frame.
        if self.vlm_coach is not None:
            self.vlm_coach.poll()
            if (self.reps > reps_before_frame and
                    self.reps % self.vlm_every_n_reps == 0):
                self.vlm_coach.submit(
                    frame, self.angle, self.reps, self.rule_cue
                )
        return frame

    def _start_rep_cycle(self, angle):
        self.cycle_started_at = time.monotonic()
        self.cycle_min_angle = angle
        self.cycle_max_angle = angle

    def _record_rep(self):
        now = time.monotonic()
        duration = max(0.1, now - self.cycle_started_at) if self.cycle_started_at else 0.0
        if self.cycle_min_angle is not None and self.cycle_max_angle is not None:
            rom = max(0.0, self.cycle_max_angle - self.cycle_min_angle)
        else:
            rom = 0.0

        # Experimental quality indicator: combines observed angular excursion
        # and rep duration. It is not a clinical form or injury-risk score.
        rom_score = min(100.0, max(0.0, (rom / 55.0) * 100.0))
        if duration < 0.8:
            tempo_score = 55.0
            tempo_note = "Very fast repetition; try a controlled pace."
        elif duration > 8.0:
            tempo_score = 75.0
            tempo_note = "Slow repetition; maintain a comfortable, steady rhythm."
        else:
            tempo_score = 95.0
            tempo_note = ""
        score = round(0.65 * rom_score + 0.35 * tempo_score, 1)
        if rom < 25:
            note = "Small measured movement range; check camera position and move only comfortably."
        elif tempo_note:
            note = tempo_note
        else:
            note = "Movement range and pace recorded."
        self.rep_events.append({
            "rep": len(self.rep_events) + 1,
            "duration": round(duration, 2),
            "rom": round(rom, 1),
            "score": score,
            "feedback": note if score < 75 else "",
        })
        self.cycle_started_at = None
        self.cycle_min_angle = None
        self.cycle_max_angle = None

    def _update_rep(self, angle):
        if self.cycle_started_at is not None:
            self.cycle_min_angle = min(self.cycle_min_angle, angle)
            self.cycle_max_angle = max(self.cycle_max_angle, angle)
        low, high = self.config["start"], self.config["finish"]
        if self.exercise == "Standing Leg Raises":
            # Lifted state is the low angle; return upright to complete.
            if self.phase in ("WAITING", "READY") and angle <= low:
                self.phase = "LIFTED"
                self._start_rep_cycle(angle)
                self.feedback = "Good lift. Lower slowly."
            elif self.phase == "LIFTED" and angle >= high:
                if time.monotonic() - self.last_rep > 0.8:
                    self.reps += 1
                    self._record_rep()
                    self.last_rep = time.monotonic()
                self.phase = "READY"
                self.feedback = "Repetition completed."
            else:
                self.feedback = "Lift your leg" if self.phase != "LIFTED" else "Lower your leg"
            return

        if self.exercise == "Seated Knee Extensions":
            if self.phase in ("WAITING", "READY") and angle <= low:
                self.phase = "BENT"
                self._start_rep_cycle(angle)
                self.feedback = "Now extend your knee."
            elif self.phase == "BENT" and angle >= high:
                self.phase = "EXTENDED"
                self.feedback = "Now bend your knee."
            elif self.phase == "EXTENDED" and angle <= low:
                if time.monotonic() - self.last_rep > 0.8:
                    self.reps += 1
                    self._record_rep()
                    self.last_rep = time.monotonic()
                self.phase = "BENT"
                self._start_rep_cycle(angle)
                self.feedback = "Repetition completed."
            else:
                self.feedback = "Bend your knee" if self.phase in ("WAITING","READY") else self.feedback
            return

        if self.exercise == "Bicep Curls":
            if self.phase in ("WAITING","READY") and angle >= high:
                self.phase = "EXTENDED"
                self._start_rep_cycle(angle)
                self.feedback = "Curl your arm."
            elif self.phase == "EXTENDED" and angle <= low:
                self.phase = "CURLED"
                self.feedback = "Extend your arm."
            elif self.phase == "CURLED" and angle >= high:
                if time.monotonic() - self.last_rep > 0.8:
                    self.reps += 1
                    self._record_rep()
                    self.last_rep = time.monotonic()
                self.phase = "EXTENDED"
                self._start_rep_cycle(angle)
                self.feedback = "Curl completed."
            else:
                self.feedback = "Extend your arm" if self.phase in ("WAITING","READY") else self.feedback
            return

        # Squats and Sit-to-Stand: low angle is down/sitting, high is up/standing.
        if self.phase in ("WAITING","READY") and angle <= low:
            self.phase = "DOWN"
            self._start_rep_cycle(angle)
            self.feedback = "Now stand up."
        elif self.phase == "DOWN" and angle >= high:
            if time.monotonic() - self.last_rep > 0.8:
                self.reps += 1
                self._record_rep()
                self.last_rep = time.monotonic()
            self.phase = "READY"
            self.feedback = "Repetition completed."
        else:
            self.feedback = "Move down" if self.phase in ("WAITING","READY") else "Stand up"

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass
        if self.vlm_coach is not None:
            self.vlm_coach.close()


# ============================================================
# CAMERA BACKENDS
# ============================================================
class WebRtcProcessor(VideoProcessorBase):
    def __init__(self, exercise, vlm_enabled=False, vlm_every_n_reps=3,
                 ollama_url=OLLAMA_URL_DEFAULT, ollama_model=OLLAMA_MODEL_DEFAULT,
                 user_code=None):
        self.workout = PoseWorkout(
            exercise, vlm_enabled, vlm_every_n_reps,
            ollama_url, ollama_model, user_code
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed = self.workout.process(img)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")


class IPStream:
    def __init__(self, url, workout):
        self.url = url
        self.workout = workout
        self.latest = None
        self.error = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        # FFmpeg is generally more reliable for MJPEG/HTTP IP Webcam streams.
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self.error = (
                "Could not open IP camera stream. Check that the phone and laptop "
                "are on the same Wi-Fi and that the URL ends in /video."
            )
            self.running = False
            return

        # Process at a capped rate. The camera can stream faster, but running
        # MediaPipe on every phone frame causes CPU overload and UI lag.
        target_interval = 0.20  # up to 5 pose-inference frames per second
        last_processed = 0.0
        consecutive_failures = 0

        while self.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_failures += 1
                self.error = "No frame received. Check the phone stream and Wi-Fi signal."
                time.sleep(0.10)
                if consecutive_failures >= 30:
                    break
                continue

            consecutive_failures = 0
            now = time.monotonic()
            if now - last_processed < target_interval:
                # Discard excess frames instead of building a processing backlog.
                continue

            try:
                # Resize before pose inference to reduce CPU usage and latency.
                h, w = frame.shape[:2]
                if w > 640:
                    frame = cv2.resize(
                        frame, (640, int(h * 640 / w)),
                        interpolation=cv2.INTER_AREA
                    )
                output = self.workout.process(frame)
                with self.lock:
                    self.latest = output
                    self.error = None
                last_processed = now
            except Exception as exc:
                self.error = f"Pose processing error: {exc}"
                time.sleep(0.1)

        cap.release()
        self.running = False

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="K-Fitness | AI Movement Coach",
                   page_icon="🏋️", layout="wide")
init_db()

# Three user-selectable visual systems. The choice is session-persistent and does
# not alter workout data, model settings, or application behaviour.
THEMES = {
    # Plum Dusk is intentionally retained exactly as the original palette.
    "Plum Dusk": {
        "bg": "#190019", "surface": "#2B124C", "surface2": "#522B5B",
        "sidebar": "#190019", "ink": "#FBE4D8", "muted": "#DFB6B2",
        "primary": "#854F6C", "primary_hover": "#9A647F", "accent": "#522B5B",
        "line": "#68466D", "hero1": "#2B124C", "hero2": "#854F6C",
        "soft": "#3A2050", "shadow": "rgba(0,0,0,.25)",
        "sidebar_ink": "#FBE4D8", "sidebar_muted": "#DFB6B2",
        "chip": "#522B5B", "chip_ink": "#FBE4D8",
        "sidebar_control": "#2B124C", "sidebar_control_ink": "#FBE4D8",
        "alert_bg": "#3A2050", "alert_border": "#68466D", "alert_ink": "#FBE4D8",
        "page_glow": "rgba(133,79,108,.16)", "primary_text": "#FFFFFF",
    },
    "Dark Green": {
        "bg": "#071B17", "surface": "#102B24", "surface2": "#183D32",
        "sidebar": "#04130F", "ink": "#E8F5EC", "muted": "#A9C4B2",
        "primary": "#62B58A", "primary_hover": "#438F6A", "accent": "#8BD6AA",
        "line": "#285442", "hero1": "#102B24", "hero2": "#1D4938",
        "soft": "#17382D", "shadow": "rgba(0,0,0,.24)",
        "sidebar_ink": "#E8F5EC", "sidebar_muted": "#A9C4B2",
        "chip": "#214B39", "chip_ink": "#E8F5EC",
        "sidebar_control": "#102B24", "sidebar_control_ink": "#E8F5EC",
        "alert_bg": "#17382D", "alert_border": "#36684F", "alert_ink": "#E8F5EC",
        "page_glow": "rgba(98,181,138,.15)", "primary_text": "#071B17",
    },
    "Minty Sage": {
        "bg": "#F1F7F2", "surface": "#FFFFFF", "surface2": "#E0EEE3",
        "sidebar": "#E6F0E8", "ink": "#203B30", "muted": "#60776A",
        "primary": "#477B60", "primary_hover": "#35634B", "accent": "#A5C9AE",
        "line": "#D2E2D6", "hero1": "#FFFFFF", "hero2": "#E0EEE3",
        "soft": "#EAF3EC", "shadow": "rgba(32,59,48,.08)",
        "sidebar_ink": "#203B30", "sidebar_muted": "#60776A",
        "chip": "#DCEBE0", "chip_ink": "#294B38",
        "sidebar_control": "#FFFFFF", "sidebar_control_ink": "#203B30",
        "alert_bg": "#E8F2EA", "alert_border": "#C7DDCD", "alert_ink": "#294B38",
        "page_glow": "rgba(165,201,174,.22)", "primary_text": "#FFFFFF",
    },
}
if st.session_state.get("kf_theme") not in THEMES:
    st.session_state.kf_theme = "Dark Green"
with st.sidebar:
    st.markdown("<div class='kf-brand-mini'>KF <span>STUDIO</span></div>", unsafe_allow_html=True)
    selected_theme = st.selectbox(
        "Appearance", list(THEMES.keys()), key="kf_theme",
        help="Choose Plum Dusk, Dark Green or Minty Sage. Your workout data is unchanged."
    )
    palette = THEMES[selected_theme]
    swatches = "".join(
        f"<span style='display:inline-block;width:25px;height:8px;border-radius:8px;background:{c};margin-right:5px'></span>"
        for c in (palette["bg"], palette["surface"], palette["primary"], palette["accent"])
    )
    st.markdown(f"<div class='kf-palette-preview'>{swatches}<small>{selected_theme} palette</small></div>", unsafe_allow_html=True)

css = """
<style>
:root {{
 --kf-bg:__BG__; --kf-surface:__SURFACE__; --kf-surface2:__SURFACE2__;
 --kf-sidebar:__SIDEBAR__; --kf-ink:__INK__; --kf-muted:__MUTED__;
 --kf-primary:__PRIMARY__; --kf-primary-hover:__PRIMARY_HOVER__; --kf-accent:__ACCENT__;
 --kf-line:__LINE__; --kf-hero1:__HERO1__; --kf-hero2:__HERO2__;
 --kf-soft:__SOFT__; --kf-shadow:__SHADOW__; --kf-sidebar-ink:__SIDEBAR_INK__;
 --kf-sidebar-muted:__SIDEBAR_MUTED__; --kf-chip:__CHIP__; --kf-chip-ink:__CHIP_INK__;
 --kf-sidebar-control:__SIDEBAR_CONTROL__; --kf-sidebar-control-ink:__SIDEBAR_CONTROL_INK__;
 --kf-alert-bg:__ALERT_BG__; --kf-alert-border:__ALERT_BORDER__; --kf-alert-ink:__ALERT_INK__;
 --kf-page-glow:__PAGE_GLOW__;
}}
.stApp {{ background:radial-gradient(ellipse at 88% 0%, var(--kf-page-glow), transparent 38%), var(--kf-bg); color:var(--kf-ink); }}
.block-container {{ padding-top:1.15rem; padding-bottom:3rem; max-width:1460px; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stToolbar"] {{ right:1rem; }}
[data-testid="stSidebar"] {{ background:var(--kf-sidebar); border-right:1px solid var(--kf-line); }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{ color:var(--kf-sidebar-ink)!important; }}
[data-testid="stSidebar"] small, [data-testid="stSidebar"] .stCaption {{ color:var(--kf-sidebar-muted)!important; }}
.kf-brand-mini {{ color:var(--kf-sidebar-ink); font-weight:850; font-size:1.05rem; letter-spacing:.08em; margin:.2rem 0 .8rem; }}
.kf-brand-mini span {{ font-size:.62rem; letter-spacing:.18em; opacity:.7; margin-left:.35rem; }}
.kf-palette-preview {{ margin:-.15rem 0 1.1rem; padding:.55rem .65rem; border:1px solid var(--kf-line); border-radius:12px; }}
.kf-palette-preview small {{ display:block; color:var(--kf-sidebar-muted); font-size:.68rem; margin-top:.25rem; }}
.kf-palette-preview {{ background:var(--kf-sidebar-control); }}
.kf-hero {{ position:relative; overflow:hidden; padding:1.65rem 1.85rem; border:1px solid var(--kf-line); border-radius:22px; background:linear-gradient(118deg,var(--kf-hero1),var(--kf-hero2)); box-shadow:0 10px 30px var(--kf-shadow); margin-bottom:1.6rem; }}
.kf-hero:after {{ content:''; position:absolute; width:230px; height:230px; right:-65px; top:-115px; border:1px solid var(--kf-primary); opacity:.12; border-radius:50%; box-shadow:0 0 0 25px transparent,0 0 0 26px var(--kf-primary),0 0 0 54px transparent,0 0 0 55px var(--kf-primary); }}
.kf-eyebrow {{ color:var(--kf-primary); text-transform:uppercase; letter-spacing:.16em; font-size:.68rem; font-weight:800; }}
.kf-hero h1 {{ color:var(--kf-ink); font-size:2.15rem; font-weight:780; margin:.28rem 0 .32rem; letter-spacing:-.055em; }}
.kf-hero p {{ color:var(--kf-muted); margin:0; font-size:.96rem; max-width:700px; }}
.kf-hero-meta {{ display:inline-flex; align-items:center; gap:.45rem; margin-top:1rem; padding:.35rem .7rem; border-radius:999px; background:var(--kf-chip); color:var(--kf-chip-ink); font-size:.72rem; font-weight:700; letter-spacing:.02em; }}
[data-testid="stMetric"] {{ background:var(--kf-surface); border:1px solid var(--kf-line); border-radius:17px; padding:16px 17px; box-shadow:0 5px 20px var(--kf-shadow); }}
[data-testid="stMetricLabel"] {{ color:var(--kf-muted)!important; font-size:.78rem; font-weight:650; }}
[data-testid="stMetricValue"] {{ color:var(--kf-primary)!important; font-weight:750; letter-spacing:-.04em; }}
[data-testid="stMetricDelta"] {{ font-size:.75rem; }}
.stButton > button, .stFormSubmitButton > button {{ border-radius:11px; border:1px solid var(--kf-line); background:var(--kf-surface); color:var(--kf-ink); font-weight:650; min-height:2.55rem; transition:all .16s ease; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{ background:var(--kf-primary); border-color:var(--kf-primary); color:__PRIMARY_BUTTON_TEXT__; }}
.stButton > button:hover {{ border-color:var(--kf-primary); color:var(--kf-primary); transform:translateY(-1px); }}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{ background:var(--kf-primary-hover); color:__PRIMARY_BUTTON_TEXT__; }}
/* Surfaces and borders */
[data-testid="stExpander"], [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stDataFrame"], [data-testid="stTable"] {{
 background:var(--kf-surface)!important;
 border:1px solid var(--kf-line)!important;
 border-radius:15px!important;
}}
[data-testid="stDataFrame"] {{ overflow:hidden; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {{
 color:var(--kf-ink)!important;
}}
/* Inputs: override BaseWeb defaults, including focus and hover outlines */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {{
 background:var(--kf-surface)!important;
 color:var(--kf-ink)!important;
 border:1px solid var(--kf-line)!important;
 border-radius:10px!important;
 box-shadow:none!important;
}}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:focus-within {{
 border-color:var(--kf-primary)!important;
 box-shadow:0 0 0 2px color-mix(in srgb, var(--kf-primary) 17%, transparent)!important;
}}
[data-testid="stSelectbox"] svg, [data-testid="stMultiSelect"] svg {{
 fill:var(--kf-muted)!important;
}}
/* BaseWeb renders select controls with nested wrappers; style every layer so
   Streamlit's default charcoal fill never leaks into a light theme. */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
 background-color:var(--kf-surface)!important;
 color:var(--kf-ink)!important;
 border-color:var(--kf-line)!important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
 background-color:var(--kf-sidebar-control)!important;
 color:var(--kf-sidebar-control-ink)!important;
}}
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stMultiSelect"] [role="combobox"] {{ color:var(--kf-ink)!important; }}
[data-testid="stSelectbox"] input,
[data-testid="stMultiSelect"] input {{ color:var(--kf-ink)!important; -webkit-text-fill-color:var(--kf-ink)!important; }}
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stMultiSelect"] [data-baseweb="select"] span {{ color:var(--kf-ink)!important; }}
[data-baseweb="popover"] [role="option"] {{ color:#24352E!important; background:#FFFFFF!important; }}
[data-baseweb="popover"] [role="option"]:hover {{ background:#EAF4EC!important; color:#18352F!important; }}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {{ color:var(--kf-muted)!important; opacity:.82!important; }}
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label,
[data-testid="stToggle"] label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{ color:var(--kf-ink); }}
/* Alerts: Streamlit's default navy info banner caused poor contrast */
[data-testid="stAlert"] {{
 background:var(--kf-alert-bg)!important;
 border:1px solid var(--kf-alert-border)!important;
 border-radius:13px!important;
 color:var(--kf-alert-ink)!important;
}}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] li,
[data-testid="stAlert"] p,
[data-testid="stAlert"] span {{
 color:var(--kf-alert-ink)!important;
}}
[data-testid="stAlert"] svg {{ fill:var(--kf-alert-ink)!important; }}
/* Buttons, including disabled and hover states */
.stButton > button, .stFormSubmitButton > button {{
 border-radius:11px;
 border:1px solid var(--kf-line)!important;
 background:var(--kf-surface)!important;
 color:var(--kf-ink)!important;
 font-weight:650;
 min-height:2.55rem;
 box-shadow:none!important;
 transition:all .16s ease;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
 border-color:var(--kf-primary)!important;
 color:var(--kf-primary)!important;
 background:var(--kf-soft)!important;
}}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
 background:var(--kf-primary)!important;
 border-color:var(--kf-primary)!important;
 color:__PRIMARY_BUTTON_TEXT__!important;
}}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
 background:var(--kf-primary-hover)!important;
 border-color:var(--kf-primary-hover)!important;
 color:__PRIMARY_BUTTON_TEXT__!important;
}}
.stButton > button:disabled, .stFormSubmitButton > button:disabled {{
 opacity:.55;
 color:var(--kf-muted)!important;
 background:var(--kf-soft)!important;
}}
/* Sidebar controls must match the sidebar rather than the main canvas */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stTextInput"] input {{
 background:var(--kf-sidebar-control)!important;
 color:var(--kf-sidebar-control-ink)!important;
 border:1px solid var(--kf-line)!important;
}}
[data-testid="stSidebar"] [data-testid="stSelectbox"] input,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] span {{
 color:var(--kf-sidebar-control-ink)!important;
}}
hr {{ border-color:var(--kf-line); }}
h1,h2,h3,h4 {{ color:var(--kf-ink); letter-spacing:-.035em; }}
small, .stCaption {{ color:var(--kf-muted); }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ color:var(--kf-sidebar-ink)!important; }}
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{ background:var(--kf-surface); }}

/* Modern dashboard layout inspired by the supplied fitness UI reference */
.kf-dashboard-heading {{ display:flex; align-items:flex-end; justify-content:space-between; gap:1rem; margin:.35rem 0 1.35rem; }}
.kf-dashboard-heading h1 {{ color:var(--kf-ink); font-size:2.15rem; letter-spacing:-.055em; margin:.25rem 0 .15rem; font-weight:800; }}
.kf-dashboard-heading p {{ color:var(--kf-muted); margin:0; }}
.kf-date-pill {{ background:var(--kf-surface); color:var(--kf-muted); border:1px solid var(--kf-line); border-radius:999px; padding:.65rem .9rem; font-size:.82rem; white-space:nowrap; }}
.kf-section-heading {{ display:flex; align-items:center; gap:.55rem; color:var(--kf-ink); font-size:1.12rem; font-weight:760; margin:1.35rem 0 .75rem; }}
.kf-section-heading span {{ display:inline-flex; align-items:center; justify-content:center; width:2rem; height:2rem; border-radius:10px; background:var(--kf-chip); color:var(--kf-primary); }}
.kf-day-tile {{ min-height:83px; padding:.6rem .2rem; border:1px solid var(--kf-line); border-radius:13px; background:var(--kf-surface); display:flex; flex-direction:column; align-items:center; justify-content:center; gap:.12rem; color:var(--kf-muted); box-shadow:0 3px 12px var(--kf-shadow); }}
.kf-day-tile small {{ font-size:.68rem; }} .kf-day-tile strong {{ color:var(--kf-ink); font-size:1.1rem; }} .kf-day-tile span {{ color:var(--kf-muted); font-size:.65rem; line-height:1; }}
.kf-day-selected {{ outline:2px solid var(--kf-primary); outline-offset:1px; }} .kf-day-done span {{ color:var(--kf-primary); }}
.kf-panel-title {{ color:var(--kf-ink); font-weight:750; font-size:1rem; padding:.25rem 0 .55rem; }}
.kf-list-row {{ display:flex; align-items:center; gap:.7rem; padding:.65rem .2rem; border-bottom:1px solid var(--kf-line); }}
.kf-list-icon {{ width:2rem; height:2rem; display:grid; place-items:center; border-radius:10px; background:var(--kf-soft); color:var(--kf-primary); }}
.kf-list-row strong {{ display:block; color:var(--kf-ink); font-size:.88rem; }} .kf-list-row small {{ display:block; color:var(--kf-muted); font-size:.73rem; margin-top:.12rem; }}
.kf-empty-panel,.kf-note-card {{ background:var(--kf-surface); border:1px solid var(--kf-line); border-radius:15px; padding:1rem; box-shadow:0 4px 16px var(--kf-shadow); color:var(--kf-ink); }}
.kf-note-card small {{ color:var(--kf-muted); }} .kf-note-card p {{ color:var(--kf-ink); margin:.5rem 0 0; font-size:.87rem; line-height:1.55; }}
.kf-notes-heading {{ margin-top:1.35rem; }} .kf-bottom-note {{ margin-top:1.6rem; padding:.8rem 1rem; border-radius:12px; background:var(--kf-soft); color:var(--kf-muted); font-size:.78rem; }}
[data-testid="stRadio"] [role="radiogroup"] {{ gap:.18rem; }}
[data-testid="stRadio"] label {{ border-radius:9px; padding:.22rem .38rem; }}
[data-testid="stRadio"] label:hover {{ background:var(--kf-sidebar-control); }}

/* Theme-aware select controls: selected value, input text and opened menu.
   BaseWeb popovers may be mounted outside the Streamlit app container, so the
   theme variables are declared on :root and remain available to the menu. */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background-color:var(--kf-surface)!important;
  border-color:var(--kf-line)!important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div *,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div *,
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stMultiSelect"] [role="combobox"],
[data-testid="stSelectbox"] input,
[data-testid="stMultiSelect"] input {
  color:var(--kf-ink)!important;
  -webkit-text-fill-color:var(--kf-ink)!important;
  opacity:1!important;
}
[data-testid="stSelectbox"] svg,
[data-testid="stMultiSelect"] svg {
  fill:var(--kf-muted)!important;
  color:var(--kf-muted)!important;
}
/* Sidebar controls use their own theme-specific foreground/background pair. */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
  background-color:var(--kf-sidebar-control)!important;
  border-color:var(--kf-line)!important;
  color:var(--kf-sidebar-control-ink)!important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div *,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"] {
  color:var(--kf-sidebar-control-ink)!important;
  -webkit-text-fill-color:var(--kf-sidebar-control-ink)!important;
  opacity:1!important;
}
/* Dropdown menu and options follow the active palette in all three themes. */
[data-baseweb="popover"],
[data-baseweb="popover"] [role="listbox"] {
  background:var(--kf-surface)!important;
  border:1px solid var(--kf-line)!important;
}
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] [role="option"] *,
[data-baseweb="popover"] [role="listbox"] * {
  color:var(--kf-ink)!important;
  -webkit-text-fill-color:var(--kf-ink)!important;
  opacity:1!important;
}
[data-baseweb="popover"] [role="option"]:hover {
  background:var(--kf-soft)!important;
}
[data-baseweb="popover"] [aria-selected="true"] {
  background:var(--kf-soft)!important;
  color:var(--kf-ink)!important;
}
/* Alerts and sidebar copy also use theme variables, not fixed white text. */
[data-testid="stAlert"],
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] *,
[data-testid="stAlert"] [data-testid="stAlertContent"] * {
  color:var(--kf-alert-ink)!important;
  -webkit-text-fill-color:var(--kf-alert-ink)!important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
  color:var(--kf-sidebar-ink)!important;
  -webkit-text-fill-color:var(--kf-sidebar-ink)!important;
}
</style>
"""
replacements = {
    "__BG__": palette["bg"], "__SURFACE__": palette["surface"],
    "__SURFACE2__": palette["surface2"], "__SIDEBAR__": palette["sidebar"],
    "__INK__": palette["ink"], "__MUTED__": palette["muted"],
    "__PRIMARY__": palette["primary"], "__PRIMARY_HOVER__": palette["primary_hover"],
    "__ACCENT__": palette["accent"], "__LINE__": palette["line"],
    "__HERO1__": palette["hero1"], "__HERO2__": palette["hero2"],
    "__SOFT__": palette["soft"], "__SHADOW__": palette["shadow"],
    "__SIDEBAR_INK__": palette["sidebar_ink"], "__SIDEBAR_MUTED__": palette["sidebar_muted"],
    "__CHIP__": palette["chip"], "__CHIP_INK__": palette["chip_ink"],
    "__SIDEBAR_CONTROL__": palette["sidebar_control"],
    "__SIDEBAR_CONTROL_INK__": palette["sidebar_control_ink"],
    "__ALERT_BG__": palette["alert_bg"], "__ALERT_BORDER__": palette["alert_border"],
    "__ALERT_INK__": palette["alert_ink"],
    "__PAGE_GLOW__": palette["page_glow"],
    "__PRIMARY_BUTTON_TEXT__": palette["primary_text"],
}
for key, value in replacements.items():
    css = css.replace(key, value)
css = css.replace("{{", "{").replace("}}", "}")
st.markdown(css, unsafe_allow_html=True)
st.markdown(
    '<div class="kf-hero"><div class="kf-eyebrow">Movement, made measurable</div>'
    '<h1>K-Fitness</h1><p>A considered space for movement practice, visual feedback and progress over time.</p>'
    '<div class="kf-hero-meta">● POSE TRACKING &nbsp;·&nbsp; VISUAL AI COACHING</div></div>',
    unsafe_allow_html=True
)

if "ip_stream" not in st.session_state:
    st.session_state.ip_stream = None
if "ip_workout" not in st.session_state:
    st.session_state.ip_workout = None
if "ip_started" not in st.session_state:
    st.session_state.ip_started = None
if "active_exercise" not in st.session_state:
    st.session_state.active_exercise = None

all_profiles = profiles()
with st.sidebar:
    st.header("Your account")
    if all_profiles:
        codes = [p["user_code"] for p in all_profiles]
        code = st.selectbox("Select profile", codes,
                            format_func=lambda x: next(
                                f'{p["name"]} ({p["user_code"]})'
                                for p in all_profiles if p["user_code"] == x))
        active = next(p for p in all_profiles if p["user_code"] == code)
    else:
        active = None
        st.info("Create a profile to begin.")
    nav_icons = {
        "Home": "⌂", "My Profile": "◉", "Workout Plan": "▦",
        "Exercises": "✚", "Workout History": "◷",
        "Progress Dashboard": "▥", "AI Coach Notes": "✎"
    }
    page = st.radio(
        "Navigation",
        list(nav_icons.keys()),
        format_func=lambda item: f"{nav_icons[item]}   {item}",
        key="main_navigation"
    )
    st.caption("Your profiles and workout records are stored locally.")

if page == "My Profile":
    st.subheader("Profile")
    existing = active or {}
    with st.form("profile_form"):
        name = st.text_input("Full name", existing.get("name", ""))
        c1,c2 = st.columns(2)
        age = c1.number_input("Age", 5, 110, int(existing.get("age", 21)))
        gender = c2.selectbox("Gender", ["Prefer not to say","Female","Male","Other"],
                              index=(["Prefer not to say","Female","Male","Other"].index(
                                  existing.get("gender") or "Prefer not to say")))
        c3,c4 = st.columns(2)
        height = c3.number_input("Height (cm)", 80.0, 250.0,
                                 float(existing.get("height_cm",165.0)), step=0.5)
        weight = c4.number_input("Weight (kg)", 20.0, 300.0,
                                 float(existing.get("weight_kg",60.0)), step=0.5)
        goal = st.selectbox("Fitness goal", GOALS,
                            index=GOALS.index(existing.get("goal",GOALS[0])))
        experience = st.selectbox("Experience", EXPERIENCE,
                                  index=EXPERIENCE.index(existing.get("experience",EXPERIENCE[0])))
        save = st.form_submit_button("Update profile" if active else "Create profile",
                                     type="primary", use_container_width=True)
    if save:
        if not name.strip():
            st.error("Enter your name.")
        else:
            data = (name.strip(), int(age),
                    None if gender == "Prefer not to say" else gender,
                    float(height), float(weight), goal, experience)
            new_code = save_profile(data, active["user_code"] if active else None)
            st.success(f"Profile saved. Your ID: {new_code}")
            st.rerun()
    if active:
        st.info(f'Your K-Fitness ID: **{active["user_code"]}**')

elif page == "Home":
    if not active:
        st.title("Welcome to K-Fitness")
        st.write("Create your profile to get started.")
    else:
        rows = get_user_sessions(active["user_code"])
        total_reps = sum(int(r["reps"] or 0) for r in rows)
        total_time = sum(float(r["duration_seconds"] or 0) for r in rows)
        today = date.today()
        st.markdown(
            f'<div class="kf-dashboard-heading"><div><div class="kf-eyebrow">YOUR MOVEMENT SPACE</div>'
            f'<h1>Good to see you, {active["name"].split()[0]}.</h1>'
            '<p>A little progress, consistently, adds up.</p></div>'
            f'<div class="kf-date-pill">◷ &nbsp; {today.strftime("%A, %d %B")}</div></div>',
            unsafe_allow_html=True
        )

        # Overview cards
        a, b, c = st.columns(3)
        a.metric("◷  Workouts completed", len(rows))
        b.metric("↗  Total repetitions", total_reps)
        c.metric("◴  Time in movement", f"{int(total_time // 60)} min")

        left, right = st.columns([1.55, 1], gap="large")
        with left:
            st.markdown('<div class="kf-section-heading"><span>▦</span> Your training calendar</div>', unsafe_allow_html=True)
            selected_day = st.date_input("Choose a date", value=today, label_visibility="collapsed", key="home_calendar_date")
            workout_days = {}
            for row in rows:
                parsed = pd.to_datetime(row.get("started_at"), errors="coerce")
                if not pd.isna(parsed):
                    day_key = parsed.date()
                    workout_days[day_key] = workout_days.get(day_key, 0) + 1
            week_start = selected_day - timedelta(days=selected_day.weekday())
            day_cols = st.columns(7)
            for offset, col in enumerate(day_cols):
                day = week_start + timedelta(days=offset)
                count = workout_days.get(day, 0)
                with col:
                    st.markdown(
                        f'<div class="kf-day-tile {"kf-day-selected" if day == selected_day else ""} '
                        f'{"kf-day-done" if count else ""}"><small>{day.strftime("%a")}</small>'
                        f'<strong>{day.day}</strong><span>{"●" if count else "·"}</span></div>',
                        unsafe_allow_html=True
                    )
            selected_count = workout_days.get(selected_day, 0)
            if selected_count:
                st.success(f"{selected_count} workout session(s) recorded on {selected_day.strftime('%d %B')}.")
            else:
                st.caption(f"No saved workout on {selected_day.strftime('%d %B')}. Your next session can start here.")

            st.markdown('<div class="kf-section-heading"><span>✚</span> Quick start</div>', unsafe_allow_html=True)
            q1, q2 = st.columns(2)
            if q1.button("✚  Start a workout", use_container_width=True, type="primary"):
                st.session_state["main_navigation"] = "Exercises"
                st.rerun()
            if q2.button("▦  View workout plan", use_container_width=True):
                st.session_state["main_navigation"] = "Workout Plan"
                st.rerun()

        with right:
            st.markdown('<div class="kf-section-heading"><span>▤</span> Your plan</div>', unsafe_allow_html=True)
            saved_plan, plan_items = get_active_plan(active["user_code"])
            if saved_plan and plan_items:
                st.markdown(f'<div class="kf-panel-title">{saved_plan["title"]}</div>', unsafe_allow_html=True)
                for item in plan_items[:4]:
                    st.markdown(
                        f'<div class="kf-list-row"><span class="kf-list-icon">✚</span>'
                        f'<div><strong>{item["exercise"]}</strong><small>{item["sets"]} sets · {item["reps_per_set"]} reps</small></div></div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown('<div class="kf-empty-panel">No plan saved yet. Create a routine that fits your goals.</div>', unsafe_allow_html=True)
                if st.button("Build my plan", use_container_width=True):
                    st.session_state["main_navigation"] = "Workout Plan"
                    st.rerun()

            st.markdown('<div class="kf-section-heading kf-notes-heading"><span>✎</span> Coach notes</div>', unsafe_allow_html=True)
            notes = get_coaching_notes(active["user_code"])
            if notes:
                latest = notes[0]
                excerpt = latest["feedback"][:230] + ("…" if len(latest["feedback"]) > 230 else "")
                st.markdown(
                    f'<div class="kf-note-card"><small>{latest["exercise"]} · {latest["created_at"]}</small>'
                    f'<p>{excerpt}</p></div>', unsafe_allow_html=True
                )
                if st.button("Open all coach notes", use_container_width=True):
                    st.session_state["main_navigation"] = "AI Coach Notes"
                    st.rerun()
            else:
                st.markdown('<div class="kf-empty-panel">Your AI movement feedback will appear here after a coached session.</div>', unsafe_allow_html=True)

        st.markdown('<div class="kf-bottom-note">K-Fitness combines pose tracking with visual AI coaching. Feedback is informational and not a medical assessment.</div>', unsafe_allow_html=True)

elif page == "Workout Plan":
    st.subheader("AI-Powered Personalized Workout Plan")

    if not active:
        st.warning("Create a profile first.")
    else:
        st.write(
            f"Creating a plan for **{active['name']}** using your saved profile: "
            f"age {active['age']}, {active['weight_kg']} kg, "
            f"{active['height_cm']} cm, goal: {active['goal']}."
        )
        st.info(
            "AI suggestions are starting points. Review and edit the plan before saving. "
            "They are not medical prescriptions."
        )

        current_plan, current_items = get_active_plan(active["user_code"])

        st.markdown("### Customize your workout requirements")
        with st.form("ai_plan_preferences_form"):
            p1, p2 = st.columns(2)
            activity_level = p1.selectbox(
                "Current activity level",
                ["Sedentary", "Lightly active", "Moderately active", "Very active"]
            )
            days_per_week = p2.selectbox(
                "Workout days per week", [2, 3, 4, 5, 6], index=1
            )

            p3, p4 = st.columns(2)
            session_minutes = p3.selectbox(
                "Preferred session duration", [15, 20, 30, 45, 60], index=2
            )
            equipment = p4.selectbox(
                "Available equipment",
                [
                    "No equipment", "Chair only", "Dumbbells",
                    "Resistance bands", "Dumbbells and resistance bands"
                ]
            )

            limitations = st.text_area(
                "Injuries, pain or physical restrictions (optional)",
                placeholder="Mention restrictions you want the plan to consider.",
                help="Avoid entering private medical information you do not want processed."
            )

            generate_ai = st.form_submit_button(
                "Generate AI Workout Plan",
                type="primary",
                use_container_width=True
            )

        if generate_ai:
            preferences = {
                "activity_level": activity_level,
                "days_per_week": days_per_week,
                "session_minutes": session_minutes,
                "equipment": equipment,
                "limitations": limitations.strip() or "None provided"
            }

            try:
                with st.spinner("Qwen is generating your personalized workout plan..."):
                    ai_plan = generate_ai_workout_plan(active, preferences)

                st.session_state["draft_plan_items"] = ai_plan["items"]
                st.session_state["draft_plan_title"] = ai_plan["title"]
                st.session_state["ai_plan_rationale"] = ai_plan["rationale"]
                st.session_state["ai_plan_frequency"] = ai_plan["weekly_frequency"]
                st.session_state["ai_plan_safety"] = ai_plan["safety_notes"]
                st.session_state["ai_plan_preferences_data"] = preferences
                st.success("Your AI-generated plan is ready. Review it below.")

            except Exception as exc:
                st.error(str(exc))
                st.caption(
                    "You can use the existing rule-based recommendation if AI is unavailable."
                )

        if st.button("Use basic recommended plan instead"):
            st.session_state["draft_plan_items"] = recommend_plan(active)
            st.session_state["draft_plan_title"] = f"{active['goal']} plan"
            for key in (
                "ai_plan_rationale", "ai_plan_frequency",
                "ai_plan_safety", "ai_plan_preferences_data"
            ):
                st.session_state.pop(key, None)

        if "draft_plan_items" not in st.session_state:
            st.session_state["draft_plan_items"] = (
                [
                    {
                        "exercise": x["exercise"],
                        "sets": x["sets"],
                        "reps": x["reps_per_set"],
                        "rest": x["rest_seconds"]
                    }
                    for x in current_items
                ]
                if current_items else recommend_plan(active)
            )

        draft = st.session_state["draft_plan_items"]

        if st.session_state.get("ai_plan_rationale"):
            st.markdown("### Why this plan?")
            st.write(st.session_state["ai_plan_rationale"])

        if st.session_state.get("ai_plan_frequency"):
            st.caption(
                "Suggested frequency: " + st.session_state["ai_plan_frequency"]
            )

        if st.session_state.get("ai_plan_safety"):
            st.warning(st.session_state["ai_plan_safety"])

        title = st.text_input(
            "Plan name",
            value=st.session_state.get(
                "draft_plan_title", f"{active['goal']} plan"
            ),
            key="plan_title_input"
        )

        st.markdown("#### Review and edit exercises, sets, repetitions and rest")
        edited = []

        for idx, item in enumerate(draft):
            with st.container(border=True):
                st.markdown(f"**Exercise {idx + 1}**")
                ex = st.selectbox(
                    "Exercise",
                    list(EXERCISES.keys()),
                    index=list(EXERCISES.keys()).index(item["exercise"]),
                    key=f"plan_ex_{idx}"
                )
                a, b, c = st.columns(3)
                sets = a.number_input(
                    "Sets", 1, 5, int(item["sets"]), key=f"plan_sets_{idx}"
                )
                reps = b.number_input(
                    "Reps per set", 1, 30, int(item["reps"]), key=f"plan_reps_{idx}"
                )
                rest = c.number_input(
                    "Rest (seconds)", 15, 300, int(item["rest"]), key=f"plan_rest_{idx}"
                )
                edited.append({
                    "exercise": ex, "sets": sets, "reps": reps, "rest": rest
                })

        add_col, remove_col = st.columns(2)
        if add_col.button("Add exercise"):
            draft.append({
                "exercise": "Squats", "sets": 2, "reps": 8, "rest": 60
            })
            st.session_state["draft_plan_items"] = draft
            st.rerun()

        if remove_col.button("Remove last exercise", disabled=len(draft) <= 1):
            draft.pop()
            st.session_state["draft_plan_items"] = draft
            st.rerun()

        if st.button(
            "Save my workout plan", type="primary", use_container_width=True
        ):
            plan_id = save_plan(
                active["user_code"],
                title.strip() or "My workout plan",
                active["goal"],
                edited
            )
            st.success(f"Workout plan saved successfully (Plan #{plan_id}).")

            for key in (
                "draft_plan_items", "draft_plan_title", "plan_title_input",
                "ai_plan_rationale", "ai_plan_frequency",
                "ai_plan_safety", "ai_plan_preferences_data"
            ):
                st.session_state.pop(key, None)
            st.rerun()

        if current_plan:
            st.divider()
            st.markdown(f"**Previously saved plan:** {current_plan['title']}")
            st.caption(f"Created {current_plan['created_at']}")
            st.dataframe(
                pd.DataFrame(current_items),
                use_container_width=True,
                hide_index=True
            )

elif page == "Exercises":
    st.subheader("Choose your exercise")
    if not active:
        st.warning("Create a profile first.")
    else:
        selected = st.selectbox("Exercise", list(EXERCISES.keys()))
        st.write(EXERCISES[selected]["description"])
        camera_mode = st.radio("Camera source",
                               ["Laptop webcam", "Phone IP camera"],
                               horizontal=True)
        ip_url = st.text_input("Phone IP camera stream URL", DEFAULT_IP_URL,
                               disabled=(camera_mode != "Phone IP camera"))
        st.caption(
            "For phone camera, keep the phone and laptop on the same Wi-Fi. "
            "Use the IP Webcam stream URL ending in /video. For smoother performance, "
            "set the phone camera to 640×480, 10 FPS and medium/low quality."
        )
        with st.expander("AI visual coach · Ollama + Qwen2.5-VL"):
            auto_vlm = st.toggle("Enable automatic VLM checkpoints", value=True,
                                 help="Analyses a selected frame after every few completed repetitions. Camera processing continues separately.")
            vlm_every_n_reps = st.select_slider(
                "Analyze after every", options=[1, 2, 3, 4, 5], value=3,
                format_func=lambda value: f"{value} rep(s)"
            )
            ollama_url = st.text_input("Ollama server URL", OLLAMA_URL_DEFAULT)
            ollama_model = st.text_input("Vision model", OLLAMA_MODEL_DEFAULT)

            st.markdown("**Live voice and review**")
            live_voice_enabled = st.toggle(
                "Speak live movement cues", value=True,
                help="Reads changed movement instructions aloud during the workout."
            )
            speak_vlm_enabled = st.toggle(
                "Read VLM feedback aloud", value=True,
                help="Speaks each newly completed Qwen2.5-VL feedback result once."
            )
            voice_review_enabled = st.toggle(
                "Enable spoken workout review", value=True,
                help="Adds a voice playback option to the written review after you finish."
            )
            if st.button("Test voice output", key="kf_test_voice"):
                speak_in_browser(
                    "K-Fitness voice coach is ready. Follow the movement cues at a comfortable pace.",
                    f"voice-test-{time.time_ns()}"
                )
            st.caption(
                "Automatic analysis runs in a background worker. Selected frames are sent "
                "to the Ollama server you configure. Voice playback uses your browser's "
                "built-in speech engine; allow audio playback in the browser if prompted."
            )

        if st.button("Start workout", type="primary", use_container_width=True):
            # Close any previous IP worker safely.
            old = st.session_state.ip_stream
            if old is not None:
                old.stop()
                st.session_state.ip_stream = None
            st.session_state.active_exercise = selected
            if camera_mode == "Phone IP camera":
                try:
                    workout = PoseWorkout(
                        selected, auto_vlm, vlm_every_n_reps,
                        ollama_url, ollama_model, active["user_code"]
                    )
                    worker = IPStream(ip_url.strip(), workout)
                    st.session_state.ip_workout = workout
                    st.session_state.ip_stream = worker
                    st.session_state.ip_started = workout.started
                    st.success("Phone IP camera workout started below.")
                except Exception as exc:
                    st.error(f"Could not start workout: {exc}")
            else:
                st.session_state.ip_workout = None
                st.session_state.ip_started = None

        if st.session_state.active_exercise:
            active_exercise = st.session_state.active_exercise
            st.divider()
            st.markdown(f"### Live workout: {active_exercise}")

            if st.session_state.ip_stream is not None:
                worker = st.session_state.ip_stream
                workout = st.session_state.ip_workout

                @st.fragment(run_every=0.35)
                def ip_camera_panel():
                    worker_now = st.session_state.ip_stream
                    if worker_now is None:
                        st.info("IP camera is stopped.")
                        return
                    if worker_now.error:
                        st.error(worker_now.error)
                    with worker_now.lock:
                        frame = None if worker_now.latest is None else worker_now.latest.copy()
                    if frame is not None:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                 channels="RGB", use_container_width=True)
                    else:
                        st.info("Connecting to phone camera...")
                    w = st.session_state.ip_workout
                    if w:
                        x,y,z = st.columns(3)
                        x.metric("Repetitions", w.reps)
                        y.metric("Phase", w.phase)
                        z.metric("Angle", f"{w.angle:.1f}°" if w.angle is not None else "--")
                        st.write(f"**Movement cue:** {w.rule_cue}")
                        if live_voice_enabled:
                            speak_in_browser(
                                w.rule_cue,
                                f"ip-{w.started:.3f}-{w.reps}-{w.phase}-{w.rule_cue}"
                            )
                        st.caption("Joint-angle cues are heuristic estimates; camera angle and visibility affect accuracy.")
                        if w.vlm_coach is not None:
                            vlm_status, vlm_note = w.vlm_coach.poll()
                            st.caption(f"VLM status: {vlm_status}")
                            if vlm_status == "Latest analysis complete":
                                st.info(vlm_note)
                                if speak_vlm_enabled:
                                    speak_in_browser(
                                        vlm_note,
                                        f"ip-vlm-{w.started:.3f}-{w.reps}-{hash(vlm_note)}"
                                    )

                ip_camera_panel()

                if st.button("Analyze current frame with AI coach", key="vlm_ip"):
                    worker_now = st.session_state.ip_stream
                    workout_now = st.session_state.ip_workout
                    with workout_now.lock:
                        snapshot = None if workout_now.latest_frame is None else workout_now.latest_frame.copy()
                    if snapshot is None:
                        st.warning("No camera frame is available yet.")
                    else:
                        with st.spinner("Analyzing one frame with the local vision model..."):
                            try:
                                note = ollama_vision_feedback(
                                    snapshot, active_exercise, workout_now.angle,
                                    workout_now.reps, workout_now.rule_cue,
                                    ollama_url, ollama_model
                                )
                                save_coaching_note(active["user_code"], active_exercise, "Local VLM", note)
                                st.session_state["latest_vlm_note"] = note
                            except Exception as exc:
                                st.error(str(exc))
                if st.session_state.get("latest_vlm_note"):
                    st.info(st.session_state["latest_vlm_note"])
                    if st.button("🔊 Listen to latest AI feedback", key="listen_latest_vlm_ip"):
                        speak_in_browser(st.session_state["latest_vlm_note"], f"manual-vlm-ip-{time.time_ns()}")

                if st.button("Finish workout", type="primary", key="finish_ip"):
                    worker.stop()
                    workout.close()
                    session_id, duration = save_workout(
                        active["user_code"], active_exercise,
                        st.session_state.ip_started, workout.reps, workout.rep_events
                    )
                    st.session_state.last_workout_review = create_workout_review(
                        active_exercise, workout.reps, duration, workout.rep_events
                    )
                    st.session_state.last_workout_review["session_id"] = session_id
                    st.session_state.ip_stream = None
                    st.session_state.ip_workout = None
                    st.session_state.ip_started = None
                    st.session_state.active_exercise = None
                    st.success(f"Workout saved! Session #{session_id} · {workout.reps} reps · {duration:.0f}s")
                    st.rerun()

            else:
                st.info("Allow camera access when your browser asks. The live camera and pose overlay will appear below.")
                ctx = webrtc_streamer(
                    key=f"kfitness-{active['user_code']}-{active_exercise}",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                    media_stream_constraints={"video": True, "audio": False},
                    video_processor_factory=lambda: WebRtcProcessor(
                        active_exercise, auto_vlm, vlm_every_n_reps,
                        ollama_url, ollama_model, active["user_code"]
                    ),
                    async_processing=True,
                )
                st.caption("Use the Stop control on the video component when you finish.")
                if ctx.video_processor:
                    w = ctx.video_processor.workout

                    @st.fragment(run_every=1.0)
                    def webcam_live_metrics(workout):
                        x,y,z = st.columns(3)
                        x.metric("Repetitions", workout.reps)
                        y.metric("Phase", workout.phase)
                        z.metric("Angle", f"{workout.angle:.1f}°" if workout.angle is not None else "--")
                        st.write(f"**Movement cue:** {workout.rule_cue}")
                        if live_voice_enabled:
                            speak_in_browser(
                                workout.rule_cue,
                                f"webcam-{workout.started:.3f}-{workout.reps}-{workout.phase}-{workout.rule_cue}"
                            )
                        st.caption("Joint-angle cues are heuristic estimates; camera angle and visibility affect accuracy.")
                        if workout.vlm_coach is not None:
                            vlm_status, vlm_note = workout.vlm_coach.poll()
                            st.caption(f"VLM status: {vlm_status}")
                            if vlm_status == "Latest analysis complete":
                                st.info(vlm_note)
                                if speak_vlm_enabled:
                                    speak_in_browser(
                                        vlm_note,
                                        f"webcam-vlm-{workout.started:.3f}-{workout.reps}-{hash(vlm_note)}"
                                    )

                    webcam_live_metrics(w)
                    if st.button("Analyze current frame with AI coach", key="vlm_laptop"):
                        with w.lock:
                            snapshot = None if w.latest_frame is None else w.latest_frame.copy()
                        if snapshot is None:
                            st.warning("No camera frame is available yet.")
                        else:
                            with st.spinner("Analyzing one frame with the local vision model..."):
                                try:
                                    note = ollama_vision_feedback(
                                        snapshot, active_exercise, w.angle, w.reps,
                                        w.rule_cue, ollama_url, ollama_model
                                    )
                                    save_coaching_note(active["user_code"], active_exercise, "Local VLM", note)
                                    st.session_state["latest_vlm_note"] = note
                                except Exception as exc:
                                    st.error(str(exc))
                    if st.session_state.get("latest_vlm_note"):
                        st.info(st.session_state["latest_vlm_note"])
                        if st.button("🔊 Listen to latest AI feedback", key="listen_latest_vlm_webcam"):
                            speak_in_browser(st.session_state["latest_vlm_note"], f"manual-vlm-webcam-{time.time_ns()}")
                    if st.button("Save and finish workout", type="primary"):
                        reps = w.reps
                        started = w.started
                        w.close()
                        sid, duration = save_workout(
                            active["user_code"], active_exercise, started, reps, w.rep_events
                        )
                        st.session_state.last_workout_review = create_workout_review(
                            active_exercise, reps, duration, w.rep_events
                        )
                        st.session_state.last_workout_review["session_id"] = sid
                        st.success(f"Workout saved! Session #{sid} · {reps} reps · {duration:.0f}s")
                        st.session_state.active_exercise = None
                        st.rerun()


        if st.session_state.get("last_workout_review"):
            review = st.session_state.last_workout_review
            st.divider()
            st.subheader("Your workout review")
            st.write(review["text"])
            if voice_review_enabled:
                if st.button("🔊 Listen to workout review", key="listen_workout_review"):
                    speak_in_browser(
                        review["text"],
                        f"workout-review-{review.get('session_id', 'latest')}"
                    )
            st.caption("The review summarizes recorded repetitions and experimental movement indicators. It is not a clinical assessment.")

elif page == "Workout History":
    st.subheader("Your workout history")
    if not active:
        st.warning("Create a profile first.")
    else:
        try:
            rows = get_user_sessions(active["user_code"])
        except Exception as exc:
            rows = []
            st.error(f"History could not be loaded: {exc}")
        if not rows:
            st.info("No workouts are currently linked to this profile.")
            with st.expander("Database diagnostics"):
                st.write(f"Database file: `{SESSION_DB}`")
                try:
                    with sqlite3.connect(SESSION_DB) as db:
                        session_count = db.execute(
                            "SELECT COUNT(*) FROM sessions"
                        ).fetchone()[0]
                        profile_count = db.execute(
                            "SELECT COUNT(*) FROM session_profiles"
                        ).fetchone()[0]
                    st.write(f"Total session records in database: **{session_count}**")
                    st.write(f"Profile links in database: **{profile_count}**")
                except Exception as exc:
                    st.error(f"Database check failed: {exc}")
        else:
            legacy_count = sum(1 for r in rows if r.get("_legacy_unlinked"))
            if legacy_count:
                st.warning(
                    f"Showing {legacy_count} older session(s) saved without a profile link. "
                    "Their original profile cannot be determined automatically."
                )
            df = pd.DataFrame(rows)
            df["Duration"] = df["duration_seconds"].apply(
                lambda v: f"{int(float(v)//60)}m {int(float(v)%60)}s"
            )
            df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce").dt.strftime("%d %b %Y, %I:%M %p")
            show = [c for c in ["id","started_at","exercise","reps","Duration","avg_form_score"] if c in df]
            st.dataframe(df[show], use_container_width=True, hide_index=True)
            st.markdown("#### Workout activity")
            history_chart = df.copy()
            history_chart["started_at"] = pd.to_datetime(history_chart["started_at"], errors="coerce")
            history_chart["reps"] = pd.to_numeric(history_chart["reps"], errors="coerce").fillna(0)
            history_chart = history_chart.dropna(subset=["started_at"]).sort_values("started_at")
            if not history_chart.empty:
                history_chart["session_label"] = history_chart.apply(
                    lambda row: f"#{int(row['id'])} · {row['exercise']}", axis=1
                )
                history_chart["latest"] = history_chart["started_at"] == history_chart["started_at"].max()
                bars = alt.Chart(history_chart).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("session_label:N", sort=None, title="Workout session",
                            axis=alt.Axis(labelAngle=-25, labelColor=palette["ink"], titleColor=palette["ink"])),
                    y=alt.Y("reps:Q", title="Repetitions",
                            axis=alt.Axis(labelColor=palette["ink"], titleColor=palette["ink"], gridColor=palette["line"])),
                    color=alt.condition("datum.latest", alt.value(palette["accent"]), alt.value(palette["primary"])),
                    tooltip=[alt.Tooltip("session_label:N", title="Session"),
                             alt.Tooltip("reps:Q", title="Repetitions"),
                             alt.Tooltip("started_at:T", title="Date")]
                )
                history_plot = (bars.properties(height=230)
                    .configure_view(stroke=palette["line"])
                    .configure_axis(domainColor=palette["line"], tickColor=palette["line"]))
                st.altair_chart(history_plot, use_container_width=True)
            st.markdown("#### Rep-by-rep movement analysis")
            chosen_session = st.selectbox(
                "Select a workout session",
                options=[int(r["id"]) for r in rows],
                format_func=lambda sid: next(
                    f"Session #{r['id']} · {r['exercise']} · {r['started_at']}"
                    for r in rows if int(r["id"]) == sid
                ),
                key="rep_detail_session"
            )
            rep_rows = get_rep_metrics(chosen_session)
            if rep_rows:
                rep_df = pd.DataFrame(rep_rows)
                rep_df = rep_df.rename(columns={
                    "rep_number": "Rep", "duration_seconds": "Duration (s)",
                    "range_of_motion": "Angular range (°)",
                    "movement_score": "Movement score", "feedback": "Observation"
                })
                st.dataframe(
                    rep_df[["Rep", "Duration (s)", "Angular range (°)",
                            "Movement score", "Observation"]],
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("This session has no rep-level metrics. New workouts will record them.")

elif page == "Progress Dashboard":
    st.subheader("Progress dashboard")
    if not active:
        st.warning("Create a profile first.")
    else:
        rows = get_user_sessions(active["user_code"])
        if not rows:
            st.info("Complete a workout to start building your progress dashboard.")
        else:
            df = pd.DataFrame(rows)
            df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
            df["reps"] = pd.to_numeric(df["reps"], errors="coerce").fillna(0)
            df["avg_form_score"] = pd.to_numeric(df["avg_form_score"], errors="coerce")
            x, y, z = st.columns(3)
            x.metric("Total workouts", len(df))
            y.metric("Total repetitions", int(df["reps"].sum()))
            z.metric("Average movement score",
                     f"{df['avg_form_score'].mean():.1f}/100"
                     if df["avg_form_score"].notna().any() else "--")
            st.markdown("#### Repetitions per workout")
            reps_chart = df.sort_values("started_at")[["started_at", "reps"]]
            reps_base = alt.Chart(reps_chart).encode(
                x=alt.X("started_at:T", title="Workout date",
                        axis=alt.Axis(labelColor=palette["ink"], titleColor=palette["ink"], grid=False)),
                y=alt.Y("reps:Q", title="Repetitions",
                        axis=alt.Axis(labelColor=palette["ink"], titleColor=palette["ink"], gridColor=palette["line"])),
                tooltip=[alt.Tooltip("started_at:T", title="Workout"),
                         alt.Tooltip("reps:Q", title="Repetitions")]
            )
            reps_plot = alt.layer(
                reps_base.mark_area(color=palette["accent"], opacity=0.22),
                reps_base.mark_line(color=palette["primary"], strokeWidth=3),
                reps_base.mark_point(color=palette["accent"], filled=True, size=85,
                                     stroke=palette["primary"], strokeWidth=1.5)
            ).properties(height=270).configure_view(stroke=palette["line"]).configure_axis(
                domainColor=palette["line"], tickColor=palette["line"]
            )
            st.altair_chart(reps_plot, use_container_width=True)
            scored = df.dropna(subset=["avg_form_score"]).sort_values("started_at")
            if not scored.empty:
                st.markdown("#### Movement consistency score over time")
                score_chart = scored[["started_at", "avg_form_score"]]
                score_base = alt.Chart(score_chart).encode(
                    x=alt.X("started_at:T", title="Workout date",
                            axis=alt.Axis(labelColor=palette["ink"], titleColor=palette["ink"], grid=False)),
                    y=alt.Y("avg_form_score:Q", title="Movement score", scale=alt.Scale(domain=[0, 100]),
                            axis=alt.Axis(labelColor=palette["ink"], titleColor=palette["ink"], gridColor=palette["line"])),
                    tooltip=[alt.Tooltip("started_at:T", title="Workout"),
                             alt.Tooltip("avg_form_score:Q", title="Movement score", format=".1f")]
                )
                score_plot = alt.layer(
                    score_base.mark_area(color=palette["accent"], opacity=0.22),
                    score_base.mark_line(color=palette["primary"], strokeWidth=3),
                    score_base.mark_point(color=palette["accent"], filled=True, size=85,
                                           stroke=palette["primary"], strokeWidth=1.5)
                ).properties(height=270).configure_view(stroke=palette["line"]).configure_axis(
                    domainColor=palette["line"], tickColor=palette["line"]
                )
                st.altair_chart(score_plot, use_container_width=True)
            st.markdown("#### Exercise-wise summary")
            summary = df.groupby("exercise").agg(
                Workouts=("id", "count"),
                Repetitions=("reps", "sum"),
                Average_score=("avg_form_score", "mean")
            ).reset_index()
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.caption(
                "Scores are experimental indicators derived from estimated joint-angle range "
                "and repetition duration. They are not clinical assessments."
            )

elif page == "AI Coach Notes":
    st.subheader("AI coach feedback history")
    if not active:
        st.warning("Create a profile first.")
    else:
        notes = get_coaching_notes(active["user_code"])
        if not notes:
            st.info("No AI coach notes yet. Start a workout and use 'Analyze current frame with AI coach'.")
        for index, note in enumerate(notes):
            with st.container(border=True):
                st.caption(f"{note['created_at']} · {note['exercise']} · {note['source']}")
                st.write(note["feedback"])
                if st.button("🔊 Listen to this feedback", key=f"listen_note_{index}"):
                    speak_in_browser(
                        note["feedback"],
                        f"saved-note-{note.get('id', index)}-{index}"
                    )

st.divider()
st.caption(
    "K-Fitness is a student project prototype. Joint-angle cues and VLM output are "
    "informational only, not a medical assessment or individualized rehabilitation prescription."
)

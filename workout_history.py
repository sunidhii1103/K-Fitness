from pathlib import Path
import sys
from datetime import date, datetime

import pandas as pd
import streamlit as st

# ------------------------------------------------------------
# PROJECT PATHS AND DATABASE IMPORT
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODULE_DIR = PROJECT_ROOT / "modules" / "module"

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from database import (  # noqa: E402
    DATABASE_PATH,
    delete_session,
    get_all_sessions,
    initialize_database,
)

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="K-Fitness | Workout History",
    page_icon="🏋️",
    layout="wide",
)

st.title("🏋️ K-Fitness Workout History")
st.caption("Your completed exercise sessions, repetition totals and workout progress.")

initialize_database()


@st.cache_data(ttl=3)
def load_sessions():
    """Load saved sessions from the project's shared SQLite database."""
    return get_all_sessions()


def format_duration(seconds):
    seconds = max(0, int(float(seconds or 0)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def safe_datetime(value):
    try:
        return pd.to_datetime(value, errors="coerce")
    except Exception:
        return pd.NaT


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
sessions = load_sessions()

st.sidebar.header("Filters")
st.sidebar.caption("Filters apply to the statistics, charts and session table.")

if not sessions:
    st.info("No workout sessions have been saved yet.")
    st.write(
        "Start an exercise from the K-Fitness exercise selector, complete a "
        "workout, and exit using the on-screen quit key. The saved session "
        "will appear here."
    )
    st.code(str(DATABASE_PATH), language="text")
    st.stop()

df = pd.DataFrame(sessions)

# Keep dates sortable even if a row contains an unexpected timestamp.
df["started_datetime"] = df["started_at"].apply(safe_datetime)
df["workout_date"] = df["started_datetime"].dt.date
df["duration_seconds"] = pd.to_numeric(
    df["duration_seconds"], errors="coerce"
).fillna(0)
df["reps"] = pd.to_numeric(df["reps"], errors="coerce").fillna(0).astype(int)
df["exercise"] = df["exercise"].fillna("Unknown").astype(str)

exercise_options = sorted(df["exercise"].unique().tolist())
selected_exercises = st.sidebar.multiselect(
    "Exercise",
    options=exercise_options,
    default=exercise_options,
)

valid_dates = df["workout_date"].dropna()
if not valid_dates.empty:
    min_date = min(valid_dates)
    max_date = max(valid_dates)
    selected_dates = st.sidebar.date_input(
        "Workout date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
else:
    selected_dates = ()

filtered = df[df["exercise"].isin(selected_exercises)].copy()

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered = filtered[
        filtered["workout_date"].between(start_date, end_date, inclusive="both")
    ]

st.sidebar.divider()
if st.sidebar.button("Refresh data", use_container_width=True):
    load_sessions.clear()
    st.rerun()

# ------------------------------------------------------------
# SUMMARY METRICS
# ------------------------------------------------------------
st.subheader("Progress overview")
if filtered.empty:
    st.warning("No sessions match the selected filters.")
    st.stop()

total_workouts = len(filtered)
total_reps = int(filtered["reps"].sum())
total_duration = filtered["duration_seconds"].sum()
active_exercises = filtered["exercise"].nunique()

metric_cols = st.columns(4)
metric_cols[0].metric("Total workouts", f"{total_workouts:,}")
metric_cols[1].metric("Total repetitions", f"{total_reps:,}")
metric_cols[2].metric("Total workout time", format_duration(total_duration))
metric_cols[3].metric("Exercises performed", f"{active_exercises:,}")

# Form scores may be absent for exercise modules that only track reps/time.
if "avg_form_score" in filtered.columns:
    form_scores = pd.to_numeric(filtered["avg_form_score"], errors="coerce")
    valid_scores = form_scores.dropna()
    if not valid_scores.empty:
        st.metric("Average recorded form score", f"{valid_scores.mean():.1f}/100")

# ------------------------------------------------------------
# CHARTS
# ------------------------------------------------------------
st.divider()
st.subheader("Workout analytics")

left, right = st.columns(2)

with left:
    st.markdown("**Repetitions by exercise**")
    reps_by_exercise = (
        filtered.groupby("exercise")["reps"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(reps_by_exercise)

with right:
    st.markdown("**Workout sessions by exercise**")
    sessions_by_exercise = (
        filtered.groupby("exercise")["id"]
        .count()
        .sort_values(ascending=False)
    )
    st.bar_chart(sessions_by_exercise)

daily = (
    filtered.dropna(subset=["workout_date"])
    .groupby("workout_date")
    .agg(workouts=("id", "count"), repetitions=("reps", "sum"))
    .sort_index()
)

if not daily.empty:
    st.markdown("**Daily workout activity**")
    st.line_chart(daily[["workouts", "repetitions"]])

# ------------------------------------------------------------
# SESSION HISTORY
# ------------------------------------------------------------
st.divider()
st.subheader("Session history")

display_df = filtered.sort_values(
    "started_datetime", ascending=False, na_position="last"
).copy()

display_df["started_at"] = display_df["started_datetime"].dt.strftime(
    "%d %b %Y, %I:%M %p"
)
display_df["duration"] = display_df["duration_seconds"].apply(format_duration)

columns_to_show = [
    "id",
    "started_at",
    "exercise",
    "reps",
    "duration",
]
if "avg_form_score" in display_df.columns:
    columns_to_show.append("avg_form_score")
if "form_corrections" in display_df.columns:
    columns_to_show.append("form_corrections")

rename_columns = {
    "id": "Session ID",
    "started_at": "Date & time",
    "exercise": "Exercise",
    "reps": "Repetitions",
    "duration": "Duration",
    "avg_form_score": "Average form score",
    "form_corrections": "Form corrections",
}

st.dataframe(
    display_df[columns_to_show].rename(columns=rename_columns),
    use_container_width=True,
    hide_index=True,
)

# ------------------------------------------------------------
# DELETE A SESSION
# ------------------------------------------------------------
with st.expander("Delete a workout session"):
    st.warning(
        "Deleting a session permanently removes that record from the local "
        "SQLite database."
    )

    session_choices = filtered.sort_values(
        "started_datetime", ascending=False, na_position="last"
    )

    choice_labels = {
        int(row["id"]): (
            f'#{int(row["id"])} — {row["exercise"]} — '
            f'{row["started_at"]} — {int(row["reps"])} reps'
        )
        for _, row in session_choices.iterrows()
    }

    selected_session_id = st.selectbox(
        "Choose the session to delete",
        options=list(choice_labels.keys()),
        format_func=lambda session_id: choice_labels[session_id],
    )

    confirm_delete = st.checkbox(
        "I understand this session will be permanently deleted."
    )

    if st.button(
        "Delete selected session",
        type="primary",
        disabled=not confirm_delete,
    ):
        delete_session(int(selected_session_id))
        load_sessions.clear()
        st.success(f"Session #{selected_session_id} deleted.")
        st.rerun()

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.divider()
st.caption(f"Database: {DATABASE_PATH}")
st.caption("K-Fitness | AI Movement Coach")

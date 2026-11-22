
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

EXERCISES = {
    "1": {
        "name": "Squats",
        "file": BASE_DIR / "pose_detector.py"
    },
    "2": {
        "name": "Seated Knee Extensions",
        "file": BASE_DIR / "exercises" / "knee_extension_camera.py"
    },
    "3": {
        "name": "Standing Leg Raises",
        "file": BASE_DIR / "exercises" / "standing_leg_raise_camera.py"
    },
    "4": {
        "name": "Sit-to-Stand",
        "file": BASE_DIR / "exercises" / "sit_to_stand_camera.py"
    },
    "5": {
        "name": "Bicep Curls",
        "file": BASE_DIR / "exercises" / "bicep_curl_camera.py"
    }
}


def display_menu():
    print("\n" + "=" * 55)
    print("             K-FITNESS AI MOVEMENT COACH")
    print("=" * 55)
    print("\nSelect an exercise:\n")

    for key, exercise in EXERCISES.items():
        print(f"  {key}. {exercise['name']}")

    print("\n  0. Exit")
    print("=" * 55)


def launch_exercise(choice):

    exercise = EXERCISES.get(choice)

    if exercise is None:
        print("\nInvalid selection.")
        return

    script_path = exercise["file"]

    if not script_path.exists():
        print("\nExercise file not found:")
        print(script_path)
        return

    print(f"\nLaunching {exercise['name']}...")

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR.parents[1]),
            check=False
        )

    except KeyboardInterrupt:
        print("\nExercise interrupted.")

    except Exception as error:
        print(f"\nError: {error}")


def main():

    while True:

        display_menu()

        choice = input("\nEnter your choice (0-5): ").strip()

        if choice == "0":
            print("\nThank you for using K-Fitness!")
            break

        launch_exercise(choice)

        input("\nPress Enter to return to the menu...")


if __name__ == "__main__":
    main()

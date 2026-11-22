
import cv2
import time
from pathlib import Path

from vlm_feedback import VLMFeedback

# Your phone IP camera
CAMERA_URL = "http://10.30.79.120:8080/video"

# Save test image in modules folder
IMAGE_PATH = Path(__file__).resolve().parent / "test_squat.jpg"

print("Connecting to K-Fitness IP camera...")

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    raise RuntimeError("Could not connect to IP camera.")

print("Camera connected!")
print("Press SPACE to capture your squat posture.")
print("Press Q to exit.")

captured = False

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not receive camera frame.")
        break

    cv2.imshow("K-Fitness - Capture Squat", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        cv2.imwrite(str(IMAGE_PATH), frame)
        captured = True
        print("Image captured!")
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if captured:

    print("\nSending image to local Qwen VLM...")

    ai = VLMFeedback()

    feedback = ai.analyze(
        image_path=str(IMAGE_PATH),
        knee_angle=110,
        reps=1,
        phase="BOTTOM"
    )

    print("\n========== K-FITNESS AI FEEDBACK ==========")
    print(feedback)
    print("===========================================")

else:
    print("No image captured.")

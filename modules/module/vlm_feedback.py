
import requests
import base64
import json


class VLMFeedback:

    def __init__(self, model="qwen2.5vl:3b"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"

    def analyze(self, image_path, knee_angle, reps, phase):

        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()
                ).decode("utf-8")

            prompt = f"""
You are the AI movement feedback assistant
for the K-Fitness physiotherapy project.

Exercise: Squat
Current knee angle: {knee_angle} degrees
Completed repetitions: {reps}
Current exercise phase: {phase}

Analyze the supplied exercise image.

Provide:
1. What you can observe about the posture.
2. One possible form issue, if visible.
3. One simple corrective suggestion.

Be concise, supportive and cautious.
Do not diagnose medical conditions.
Do not claim to measure angles from the image.
If posture is unclear, say so.
"""

            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_data]
                    }
                ],
                "stream": False
            }

            response = requests.post(
                self.url,
                json=payload,
                timeout=120
            )

            response.raise_for_status()

            result = response.json()

            return result["message"]["content"]

        except requests.exceptions.ConnectionError:
            return "Ollama is not running. Please start Ollama."

        except requests.exceptions.Timeout:
            return "AI analysis timed out. Try again."

        except Exception as error:
            return f"AI feedback error: {error}"

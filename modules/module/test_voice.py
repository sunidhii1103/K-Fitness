
import pyttsx3

print("Initializing voice engine...")

engine = pyttsx3.init("sapi5")

voices = engine.getProperty("voices")

print("Available voices:", len(voices))

engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

print("Testing speech...")

engine.say("Hello! This is K Fitness. Your voice coach is working.")

engine.runAndWait()

print("Voice test finished.")

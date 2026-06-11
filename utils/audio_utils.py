import pyttsx3
import time

engine = pyttsx3.init()
last_spoken = ""
last_time = 0

def speak_once(text, cooldown=2):
    global last_spoken, last_time

    now = time.time()

    if text != last_spoken or (now - last_time) > cooldown:
        engine.say(text)
        engine.runAndWait()
        last_spoken = text
        last_time = now
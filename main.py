#!/usr/bin/env python3
"""
AI Personal Guide for Visually Challenged
Hardware: RPi 4B, OV5647 Camera, HC-SR04 Ultrasonic, AUX Audio
"""

import cv2
import numpy as np
import onnxruntime as ort
import RPi.GPIO as GPIO
import subprocess
import time

# ─── GPIO SETUP ──────────────────────────────────────────────
TRIG = 23
ECHO = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# ─── COCO CLASS NAMES ────────────────────────────────────────
# Filtered classes relevant to indoor navigation
# Index must match original COCO positions
CLASSES = {
    0: "person",
    13: "bench",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    42: "fork",
    43: "knife",
    45: "bowl",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    66: "keyboard",
    67: "cell phone",
    69: "oven",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    76: "scissors",
    77: "teddy bear",
}
# ─── AUDIO ───────────────────────────────────────────────────
def speak(text):
    print(text)
    subprocess.Popen(
        ["espeak", "-v", "en", "-s", "140", "-a", "200", text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# ─── ULTRASONIC ──────────────────────────────────────────────
def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.02)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    timeout = time.time() + 0.04
    while GPIO.input(ECHO) == 0:
        if time.time() > timeout:
            return None
    pulse_start = time.time()

    timeout = time.time() + 0.04
    while GPIO.input(ECHO) == 1:
        if time.time() > timeout:
            return None
    pulse_end = time.time()

    distance_m = round(((pulse_end - pulse_start) * 34300) / 200, 1)
    return distance_m if 0.02 < distance_m < 4.0 else None

# ─── DIRECTION ───────────────────────────────────────────────
def get_direction(cx_ratio, cy_ratio):
    """
    cx_ratio: center x of bounding box as ratio of frame width  (0.0 to 1.0)
    cy_ratio: center y of bounding box as ratio of frame height (0.0 to 1.0)
    Returns a simple navigation instruction.
    """
    if cy_ratio < 0.25:
        return "overhead"
    if cx_ratio < 0.30:
        return "move right"
    elif cx_ratio > 0.70:
        return "move left"
    else:
        return "ahead"

# ─── PREPROCESS ──────────────────────────────────────────────
def preprocess(frame, size=320):
    img = cv2.resize(frame, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

# ─── DETECT ──────────────────────────────────────────────────
def detect(session, frame, conf_thresh=0.45):
    blob = preprocess(frame)
    outputs = session.run(None, {"images": blob})[0]

    # YOLOv8 output: [1, 84, 2100] → transpose to [2100, 84]
    outputs = np.squeeze(outputs).T

    results = {}  # label -> (confidence, direction)

    for row in outputs:
        scores = row[4:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if confidence > conf_thresh:
            label = CLASSES[class_id]

            # Bounding box center normalized to 320x320 input
            cx = float(row[0]) / 320.0
            cy = float(row[1]) / 320.0
            direction = get_direction(cx, cy)

            # Keep only highest confidence per class
            if label not in results or confidence > results[label][0]:
                results[label] = (confidence, direction)

    return [(label, info[1]) for label, info in results.items()]

# ─── MAIN ────────────────────────────────────────────────────
def main():
    print("=== AI Guide System Starting ===")

    # Load ONNX model
    print("[MODEL] Loading YOLOv8n ONNX...")
    session = ort.InferenceSession(
        "yolov8n.onnx",
        providers=["CPUExecutionProvider"]
    )
    print("[MODEL] Ready")

    # Init camera
    print("[CAM] Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("[ERROR] Camera not found!")
        GPIO.cleanup()
        return
    print("[CAM] Ready")

    speak("Guide system ready")

    last_spoken = {}
    COOLDOWN = 5  # seconds between repeating same object

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame capture failed, retrying...")
                time.sleep(0.5)
                continue

            # Get distance from ultrasonic
            dist = get_distance()
            dist_str = f"{dist} metres" if dist else "unknown distance"

            # Run object detection
            detected = detect(session, frame)

            now = time.time()
            for label, direction in detected:
                if now - last_spoken.get(label, 0) > COOLDOWN:

                    # Urgency based on distance
                    if dist and dist < 0.5:
                        msg = f"stop, {label} very close, {dist_str}"
                    elif dist and dist < 1.0:
                        msg = f"caution, {label}, {dist_str}, {direction}"
                    else:
                        msg = f"object detected, {label}, {dist_str}, {direction}"

                    speak(msg)
                    last_spoken[label] = now

            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n[STOP] Shutting down")
        speak("Shutting down")
    finally:
        cap.release()
        GPIO.cleanup()

if __name__ == "__main__":
    main()

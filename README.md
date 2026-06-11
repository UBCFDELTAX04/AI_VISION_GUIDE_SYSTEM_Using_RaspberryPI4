# AI-Based Personal Navigation Assistant for the Visually Impaired

> A real-time, fully offline assistive navigation system running on a Raspberry Pi 4B — combining computer vision, ultrasonic sensing, and text-to-speech audio to guide visually impaired users through their environment.

---

## Table of Contents

- [Overview](#overview)
- [Demo Output](#demo-output)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Stack](#software-stack)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Auto-Start on Boot](#auto-start-on-boot)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

This project implements a standalone assistive navigation device for visually impaired individuals. It runs entirely on a **Raspberry Pi 4B** with no desktop environment, no cloud dependency, and no internet connection required at runtime.

The system:
- Detects objects in real time using a **YOLOv8n ONNX model**
- Measures proximity using an **HC-SR04 ultrasonic sensor**
- Determines the **spatial direction** of detected objects (ahead, move left, move right, overhead)
- Delivers **spoken audio alerts** through an AUX speaker via text-to-speech
- Logs all detections to a persistent log file
- Boots and runs **automatically on power-on** via systemd

---

## Demo Output

```
=== AI Guide System Starting ===
[MODEL] Loading YOLOv8n ONNX...
[MODEL] Ready
[CAM] Opening camera...
[CAM] Ready
Guide system ready

object detected, person, 1.9 metres, ahead
caution, chair, 0.9 metres, move right
stop, bottle, 0.4 metres, ahead
object detected, laptop, 2.1 metres, move left
```

All output is simultaneously spoken aloud through the AUX audio output.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Raspberry Pi 4B                       │
│                                                         │
│  ┌──────────────┐     ┌─────────────────────────────┐  │
│  │  OV5647 Cam  │────►│  OpenCV Frame Capture       │  │
│  └──────────────┘     │  640×480 BGR                │  │
│                        └────────────┬────────────────┘  │
│  ┌──────────────┐                   │                   │
│  │  HC-SR04     │────►  Distance    │ Preprocess        │
│  │  Ultrasonic  │      (metres)     │ Resize 320×320    │
│  └──────────────┘          │        │ Normalize 0–1     │
│                             │        ▼                   │
│                             │  ┌────────────────────┐   │
│                             │  │  YOLOv8n ONNX      │   │
│                             │  │  ONNX Runtime CPU  │   │
│                             │  │  2100 candidates   │   │
│                             │  └────────┬───────────┘   │
│                             │           │               │
│                             │  Filter conf > 0.45       │
│                             │  Get direction from cx/cy │
│                             │           │               │
│                             └──────────►│               │
│                                         ▼               │
│                             ┌───────────────────────┐   │
│                             │  Build spoken message │   │
│                             │  Apply cooldown timer │   │
│                             └───────────┬───────────┘   │
│                                         │               │
│                    ┌────────────────────┴─────────┐     │
│                    ▼                              ▼     │
│             espeak → aplay               guide.log      │
│             AUX audio output             Terminal       │
└─────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

| Component | Specification |
|---|---|
| Single-board computer | Raspberry Pi 4B (2GB RAM) |
| Camera | Raspberry Pi Camera Module v1 (OV5647) |
| Distance sensor | HC-SR04 Ultrasonic Sensor |
| Audio output | 3.5mm AUX speaker or headphones |
| Storage | MicroSD card (32GB recommended) |
| Power | 5V 3A USB-C power supply |

### Wiring — HC-SR04 to Raspberry Pi GPIO

```
HC-SR04 VCC  →  Pin 2  (5V)
HC-SR04 GND  →  Pin 6  (GND)
HC-SR04 TRIG →  Pin 16 (GPIO 23)
HC-SR04 ECHO →  Pin 18 (GPIO 24)  ← via voltage divider (1kΩ + 2kΩ) to 3.3V
```

> ⚠️ The ECHO pin outputs 5V. The RPi GPIO tolerates only 3.3V. Always use a voltage divider on the ECHO line to avoid damaging the Pi.

---

## Software Stack

| Component | Technology |
|---|---|
| OS | Debian Trixie Lite 64-bit (headless) |
| Language | Python 3.13 |
| Object detection | YOLOv8n — ONNX format |
| Inference engine | ONNX Runtime 1.26.0 (CPU) |
| Camera interface | OpenCV (`/dev/video0`) |
| Distance sensing | RPi.GPIO |
| Text-to-speech | espeak piped to aplay |
| Service management | systemd |
| Virtual environment | Python venv (`--system-site-packages`) |

### Why ONNX Runtime instead of PyTorch?

Installing `ultralytics` with PyTorch on the Raspberry Pi pulls in PyTorch (~419MB) and CUDA dependencies (~433MB), exhausting the SD card and RAM. ONNX Runtime is only 16MB and runs the same model with identical accuracy using CPU inference only — the correct approach for edge deployment.

---

## How It Works

### 1. Distance Measurement (HC-SR04)

The Pi sends a 10-microsecond pulse on the TRIG pin. The sensor fires 8 ultrasonic bursts at 40kHz and raises the ECHO pin HIGH until the reflection returns. The Pi measures the ECHO duration and converts it to distance:

```
distance (m) = (echo_duration_seconds × 34300) / 200
```

Valid range: 0.02m to 4.0m. Readings outside this range are discarded.

### 2. Frame Capture and Preprocessing

OpenCV captures a 640×480 BGR frame from `/dev/video0`. The frame is:
- Resized to 320×320 (fixed model input size)
- Converted from BGR to RGB
- Normalized from pixel range 0–255 to 0.0–1.0
- Reshaped to a 4D tensor: `[1, 3, 320, 320]`

### 3. Object Detection (YOLOv8n ONNX)

The tensor is passed to ONNX Runtime. YOLOv8n processes it through:
- **Backbone (CSPNet)** — extracts visual features at multiple scales
- **Neck (PAN-FPN)** — merges small, medium, and large-scale features
- **Detection Head** — outputs `[1, 84, 2100]`: 2100 candidate boxes, each with 4 geometry values and 80 class confidence scores

For each candidate:
```python
class_id   = argmax(scores[4:])      # highest scoring class
confidence = scores[4 + class_id]    # confidence value
keep if confidence > 0.45            # filter threshold
```

### 4. Spatial Direction

The bounding box center coordinates (`cx`, `cy`) are normalized to 0–1 ratios and mapped to navigation instructions:

```
cx < 0.30              → "move right"   (object is on your left)
cx > 0.70              → "move left"    (object is on your right)
0.30 ≤ cx ≤ 0.70      → "ahead"
cy < 0.25              → "overhead"
```

### 5. Urgency Tiers

Distance from the ultrasonic sensor determines the urgency level of the spoken message:

| Distance | Message format |
|---|---|
| < 0.5m | `stop, [object] very close, [distance]` |
| 0.5m – 1.0m | `caution, [object], [distance], [direction]` |
| > 1.0m | `object detected, [object], [distance], [direction]` |

### 6. Cooldown System

To prevent audio overload, each object class has an independent cooldown timer:

| Urgency | Repeat interval |
|---|---|
| Stop | Every 3 seconds |
| Caution | Every 6 seconds |
| Normal | Every 10 seconds |

### 7. Audio Output

Speech is generated by `espeak` and piped to `aplay` targeting the onboard 3.5mm AUX output:

```bash
espeak --stdout "message" | aplay -D plughw:0,0
```

---

## Installation

### Step 1 — Flash OS and enable SSH

Flash **Debian Trixie Lite 64-bit** to your SD card using Raspberry Pi Imager. Enable SSH during setup.

### Step 2 — Initial system configuration

```bash
# Connect via SSH
ssh pi9@<your-rpi-ip>

# Create swap file (2GB)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Reduce GPU memory (no display needed)
echo 'gpu_mem=16' | sudo tee -a /boot/firmware/config.txt

sudo reboot
```

### Step 3 — Install system dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv espeak libespeak-dev \
    python3-picamera2 libopencv-dev python3-opencv \
    python3-rpi.gpio git libopenblas-dev
```

### Step 4 — Clone repository and set up environment

```bash
git clone https://github.com/yourusername/ai-guide.git ~/guide_project
cd ~/guide_project

python3 -m venv venv_pi --system-site-packages
source venv_pi/bin/activate

pip install --no-cache-dir onnxruntime RPi.GPIO gpiozero
```

### Step 5 — Add the ONNX model

Export `yolov8n.onnx` on a Windows/Linux PC with sufficient resources:

```python
from ultralytics import YOLO
YOLO("yolov8n.pt").export(format="onnx", imgsz=320, simplify=True)
```

Then copy to the Pi:

```bash
scp yolov8n.onnx pi9@<your-rpi-ip>:~/guide_project/
```

### Step 6 — Verify hardware

```bash
source venv_pi/bin/activate

# Test camera
python3 -c "import cv2; cap=cv2.VideoCapture(0); ret,f=cap.read(); print('Camera OK:', f.shape); cap.release()"

# Test audio
espeak --stdout "System ready" | aplay -D plughw:0,0

# Test model
python3 -c "import onnxruntime as ort; s=ort.InferenceSession('yolov8n.onnx'); print('Model OK:', s.get_inputs()[0].shape)" 2>/dev/null
```

---

## Project Structure

```
guide_project/
├── main.py                 # Main application script
├── yolov8n.onnx            # YOLOv8n ONNX model (13MB)
├── start.sh                # Service launch script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── logs/
│   └── guide.log           # Runtime detection log
├── venv_pi/                # Python virtual environment (not committed)
├── config/                 # Configuration files
├── data/                   # Calibration and sample data
├── models/                 # Alternative model storage
└── scripts/                # Utility scripts
```

---

## Configuration

Key parameters in `main.py`:

```python
# GPIO pins
TRIG = 23                   # Ultrasonic trigger pin (GPIO 23)
ECHO = 24                   # Ultrasonic echo pin (GPIO 24)

# Detection threshold
conf_thresh = 0.45          # Minimum confidence to report an object

# Cooldown timers (seconds)
COOLDOWN_STOP    = 3        # Repeat "stop" alerts every 3s
COOLDOWN_CAUTION = 6        # Repeat "caution" alerts every 6s
COOLDOWN_NORMAL  = 10       # Repeat normal detections every 10s

# Urgency distance thresholds
STOP_DISTANCE    = 0.5      # metres — triggers "stop"
CAUTION_DISTANCE = 1.0      # metres — triggers "caution"
```

---

## Running the System

### Manual run

```bash
cd ~/guide_project
source venv_pi/bin/activate
python3 main.py
```

### Watch live logs (when running as service)

```bash
tail -f ~/guide_project/logs/guide.log
```

---

## Auto-Start on Boot

### 1. Create the systemd service

```bash
sudo nano /etc/systemd/system/guide.service
```

```ini
[Unit]
Description=AI Visual Guide System
After=network.target multi-user.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=pi9
WorkingDirectory=/home/pi9/guide_project
Environment=ALSA_CARD=0
Environment=XDG_RUNTIME_DIR=/run/user/1000
ExecStartPre=/bin/sleep 10
ExecStart=/home/pi9/guide_project/venv_pi/bin/python3 /home/pi9/guide_project/main.py
Restart=always
RestartSec=10
StandardOutput=append:/home/pi9/guide_project/logs/guide.log
StandardError=append:/home/pi9/guide_project/logs/guide.log

[Install]
WantedBy=multi-user.target
```

### 2. Enable the service

```bash
sudo systemctl daemon-reload
sudo systemctl enable guide.service
sudo systemctl start guide.service
```

### 3. Useful service commands

```bash
sudo systemctl status guide.service     # Check if running
sudo systemctl restart guide.service    # Restart
sudo systemctl stop guide.service       # Stop
sudo systemctl disable guide.service    # Disable autostart
```

---

## Known Limitations

- **Single ultrasonic sensor** — measures distance in one direction only (forward). Objects to the side are detected visually but distance is approximate.
- **YOLOv8n accuracy** — the nano model trades accuracy for speed. False detections can occur in cluttered or low-light environments.
- **~3 FPS inference rate** — adequate for slow walking pace but not fast movement.
- **Fixed 320×320 input** — re-export the ONNX model to change resolution.
- **Indoor-optimised class filter** — 28 of 80 COCO classes are active. Outdoor classes (vehicles, traffic signs) are currently filtered out.

---

## Future Improvements

- [ ] Custom-trained model on indoor navigation dataset for higher accuracy
- [ ] Multiple ultrasonic sensors (left, right, forward) for full spatial awareness
- [ ] Depth estimation from monocular camera to supplement ultrasonic
- [ ] Wake word detection to toggle the system on/off by voice
- [ ] Battery level monitoring and low-battery audio alert
- [ ] GPS integration for outdoor navigation mode
- [ ] Mobile companion app for caregiver monitoring

---

## Built With

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection model
- [ONNX Runtime](https://onnxruntime.ai/) — lightweight inference engine
- [OpenCV](https://opencv.org/) — camera capture and image processing
- [espeak](http://espeak.sourceforge.net/) — text-to-speech engine
- [RPi.GPIO](https://pypi.org/project/RPi.GPIO/) — GPIO control for ultrasonic sensor

---

## Author

**Aditya**
- Built as an assistive technology project for visually impaired individuals
- Developed and deployed on Raspberry Pi 4B running Debian Trixie 64-bit

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

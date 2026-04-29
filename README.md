# Computer Vision Project

A project for gesture recognition using Mediapipe, OpenCV, and neural networks. This repository includes scripts for live video hand gesture recognition and tools for training a model on custom gestures.

---

## Features
- **Live Gesture Recognition**: Use `mediapipeLive.py` for real-time hand gesture detection.
- **Custom Gesture Training**: Train a neural network on labeled landmarks using Mediapipe.

---

## Creating a Virtual Environment (venv)

### Linux/macOS
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows
```cmd
python -m venv venv
venv\Scripts\activate
```

---

## How to Run Live Gesture Recognition
1. Clone this repository and navigate to its directory:
   ```bash
   git clone <repository_url>
   cd Computer-Vision-Project
2. Ensure all dependencies are installed. You may use pip to install required libraries:
   ```bash
   pip install -r requirements.txt

3. run mediapipeLive.py
   ```bash
   python3 mediapipeLive.py

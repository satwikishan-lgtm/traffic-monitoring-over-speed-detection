# Real-Time Traffic Monitoring and Over-Speed Detection

Detects and tracks vehicles with **YOLOv8** + **OpenCV**, estimates each vehicle's speed, and raises
a **visual alert** (red box, X, banner) and an **audio alert** (`assets/beep.wav`, a 1000 Hz beep) on over-speeding.

## Project layout
```
main.py              run this: video loop, drawing, alerts, CSV log
detector.py          YOLOv8 detection + tracking wrapper
speed_estimator.py   pixel movement -> km/h (windowed + smoothed)
alerts.py            non-blocking audio alert, works on Windows/macOS/Linux
models/yolov8n.pt    YOLOv8 nano weights
assets/              beep.wav (alert sound)
videos/              three sample videos
tests/test_logic.py  tests (no GPU / model needed)
```

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## Run
```bash
python main.py                                                  # sample video 1, live window
python main.py --source videos/sample_traffic_2.mp4
python main.py --source videos/sample_traffic_3.mp4
python main.py --max-frames 300                                 # quick test (like your notebook)
python main.py --save output/result.mp4 --log output/violations.csv
python main.py --source 0                                       # webcam
```
Press **ESC** or **q** to quit. All options: `python main.py --help`.

| Option | Default | Meaning |
|---|---|---|
| `--limit` | 60 | km/h above which the vehicle is flagged (visual alert) |
| `--alarm-speed` | 80 | km/h above which the sound also plays |
| `--ppm` | 10 | calibration: pixels per metre |
| `--window` | 0.5 | seconds of movement used for each speed reading |
| `--classes` | car truck bus motorcycle bicycle cow | what to track |

## Calibration (important)
Speed = pixel distance / `--ppm` / time. The default `--ppm 10` is a placeholder, so the km/h values are
only meaningful after you calibrate for your camera:
1. Pick something in the scene with a known real length (a lane is about 3.5 m wide, a car about 4.5 m long).
2. Measure it in pixels in a video frame.
3. `ppm = pixels / metres`, then run with `--ppm <value>`.

A single value is only accurate near the distance where you measured it, because objects further from the
camera cover fewer pixels per metre. For the best results, use a camera looking straight along the road
and measure on the road surface where vehicles actually pass.

## Changes from the notebook version
- **`motorbike` to `motorcycle`**: the COCO label is `motorcycle`, so motorbikes were never detected before.
  `bus` added.
- **Colour fix**: frames were converted to RGB before going into YOLO, which expects OpenCV's BGR. Removed.
- **Sound**: `winsound.Beep` froze the video for 0.5 s per frame and only works on Windows. Now plays your WAV
  asynchronously with a 2 s cooldown, on any OS.
- **Steadier speeds**: measured over a 0.5 s window and smoothed, instead of frame-to-frame jumps.
- Speed is taken from the bottom-centre of the box (where the vehicle touches the road) instead of the box centre.
- Old tracks are cleaned up, a webcam FPS of 0 is handled, and there are CLI options, video saving and a CSV log.

## Tests
```bash
python tests/test_logic.py
```

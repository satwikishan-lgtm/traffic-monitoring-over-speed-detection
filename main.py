"""Real-time traffic monitoring and over-speed detection (YOLOv8 + OpenCV).

Examples
    python main.py                                   # sample video, live window
    python main.py --source videos/sample_traffic_3.mp4 --save output/result.mp4
    python main.py --source 0                        # webcam
    python main.py --max-frames 300 --limit 40       # quick test, lower speed limit
Press ESC or q in the video window to quit.
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

import cv2

from alerts import AudioAlert
from detector import DEFAULT_CLASSES
from speed_estimator import SpeedEstimator

BASE = Path(__file__).resolve().parent
GREEN, RED, WHITE, BLACK = (0, 200, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)  # BGR


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=str(BASE / "videos" / "sample_traffic_1.mp4"),
                   help="video file path, or a camera index such as 0")
    p.add_argument("--weights", default=str(BASE / "models" / "yolov8n.pt"))
    p.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    p.add_argument("--conf", type=float, default=0.3, help="detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tracker", default="botsort.yaml", help="botsort.yaml or bytetrack.yaml")
    p.add_argument("--limit", type=float, default=60.0, help="over-speed limit in km/h (visual alert)")
    p.add_argument("--alarm-speed", type=float, default=80.0, help="speed in km/h that also plays the sound")
    p.add_argument("--ppm", type=float, default=10.0, help="calibration: pixels per metre")
    p.add_argument("--window", type=float, default=0.5, help="seconds of history used for each speed reading")
    p.add_argument("--sound", default=str(BASE / "assets" / "beep.wav"))
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--no-display", action="store_true", help="do not open a video window")
    p.add_argument("--save", default="", help="write the annotated video to this path")
    p.add_argument("--log", default="", help="write over-speed events to this CSV file")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = whole video)")
    return p.parse_args(argv)


def draw_label(img, text, x, y, color, scale):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    y = max(y, h + base + 2)
    cv2.rectangle(img, (x, y - h - base - 2), (x + w + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 2, cv2.LINE_AA)


def annotate(frame, det, speed, limit):
    x1, y1, x2, y2 = det.box
    scale = max(0.45, frame.shape[1] / 1600)
    over = speed > limit
    color = RED if over else GREEN
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, ((x1 + x2) // 2, y2), 4, color, -1)       # point used for speed
    draw_label(frame, f"{det.label} #{det.track_id} {int(speed)} km/h", x1, y1 - 4, color, scale)
    if over:
        cv2.line(frame, (x1, y1), (x2, y2), RED, 2)
        cv2.line(frame, (x1, y2), (x2, y1), RED, 2)
        draw_label(frame, "Over-speed!", x1, y2 + 20, RED, scale)


def run(args, detector=None):
    """Process the video. `detector` is any callable frame -> [Detection]; defaults to YOLOv8."""
    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Cannot open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps < 1:
        fps = 30.0                                              # webcams often report 0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if detector is None:
        from detector import YoloTracker
        detector = YoloTracker(args.weights, args.classes, args.conf, args.imgsz, args.tracker)

    estimator = SpeedEstimator(fps, args.ppm, window_seconds=args.window)
    audio = AudioAlert(args.sound, enabled=not args.no_audio)

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    violations = {}                                             # track_id -> event info
    frame_idx, t0 = 0, time.time()
    print(f"Source: {args.source}  ({width}x{height} @ {fps:.1f} fps)  limit={args.limit} km/h")

    try:
        while not (args.max_frames and frame_idx >= args.max_frames):
            ok, frame = cap.read()
            if not ok:
                print("End of video.")
                break

            over_speed = alarm = False
            for det in detector(frame):
                x1, y1, x2, y2 = det.box
                speed = estimator.update(det.track_id, ((x1 + x2) / 2, y2), frame_idx)
                annotate(frame, det, speed, args.limit)

                if speed > args.limit:
                    over_speed = True
                    ev = violations.setdefault(det.track_id, {
                        "track_id": det.track_id, "class": det.label,
                        "first_frame": frame_idx, "time_s": round(frame_idx / fps, 2), "peak_kmh": 0.0})
                    ev["peak_kmh"] = max(ev["peak_kmh"], round(speed, 1))
                if speed > args.alarm_speed:
                    alarm = True

            estimator.cleanup(frame_idx)
            if over_speed:
                cv2.putText(frame, "ALERT: Over-speeding Detected!", (30, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, max(0.7, width / 1200), RED, 3, cv2.LINE_AA)
            if alarm:
                audio.trigger()

            if writer:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Traffic Monitoring - ESC/q to quit", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = max(time.time() - t0, 1e-6)
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed:.1f} fps). "
          f"Over-speed vehicles: {len(violations)}")
    events = sorted(violations.values(), key=lambda e: e["first_frame"])
    if args.log and events:
        Path(args.log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(events[0].keys()))
            w.writeheader()
            w.writerows(events)
        print(f"Violation log saved to {args.log}")
    return events


if __name__ == "__main__":
    run(parse_args())

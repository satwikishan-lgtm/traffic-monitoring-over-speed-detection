"""Tests that run without ultralytics/torch, using a fake detector.
Run from the project folder:  python tests/test_logic.py   (or: pytest)
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as app
from alerts import AudioAlert
from detector import Detection
from speed_estimator import SpeedEstimator

ROOT = Path(__file__).resolve().parent.parent
VIDEO = str(ROOT / "videos" / "sample_traffic_2.mp4")       # 480x272, 25 fps, 150 frames


def test_constant_speed_is_measured_correctly():
    fps, ppm = 25, 10
    est = SpeedEstimator(fps, ppm)
    # 20 px/frame -> 500 px/s -> 50 m/s -> 180 km/h
    speed = 0
    for i in range(30):
        speed = est.update(1, (20 * i, 100), i)
    assert abs(speed - 180) < 1.0, speed


def test_stationary_object_is_zero_and_stale_tracks_dropped():
    est = SpeedEstimator(25, 10)
    for i in range(20):
        assert est.update(7, (50, 50), i) == 0
    est.cleanup(200)
    assert 7 not in est._history


def test_first_frames_report_zero():
    est = SpeedEstimator(25, 10)
    assert est.update(1, (0, 0), 0) == 0
    assert est.update(1, (50, 0), 1) == 0


def test_audio_cooldown():
    a = AudioAlert(None, cooldown=100, enabled=True)
    a._play = lambda: None
    assert a.trigger() is True
    assert a.trigger() is False


def test_full_pipeline_with_fake_detector():
    # One vehicle moving 6 px/frame at 25 fps, ppm=10 -> 15 m/s -> 54 km/h
    def fake(frame):
        fake.i += 1
        x = 20 + 6 * fake.i
        return [Detection(1, "car", (x, 100, x + 60, 160), 0.9)]
    fake.i = -1

    with tempfile.TemporaryDirectory() as d:
        out, log = f"{d}/out.mp4", f"{d}/log.csv"
        args = app.parse_args(["--source", VIDEO, "--no-display", "--no-audio",
                               "--limit", "40", "--save", out, "--log", log, "--max-frames", "60"])
        events = app.run(args, detector=fake)
        assert Path(out).stat().st_size > 1000
        assert len(events) == 1 and events[0]["class"] == "car"
        assert 50 <= events[0]["peak_kmh"] <= 58, events[0]
        assert Path(log).exists()

        # Above the limit of 60 the same car must NOT be flagged
        fake.i = -1
        args = app.parse_args(["--source", VIDEO, "--no-display", "--no-audio",
                               "--limit", "60", "--max-frames", "60"])
        assert app.run(args, detector=fake) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)

"""Speed estimation from tracked object positions.

Speed is measured over a short time window (default 0.5 s) instead of between
two consecutive frames, which removes most of the jitter from bounding-box
noise, and is then smoothed with an exponential moving average.
"""
import math
from collections import deque


class SpeedEstimator:
    def __init__(self, fps, pixels_per_meter=10.0, window_seconds=0.5,
                 smoothing=0.4, min_frames=3, max_missing_frames=30):
        if pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be positive")
        self.fps = float(fps)
        self.pixels_per_meter = float(pixels_per_meter)
        self.window_frames = max(min_frames, round(window_seconds * self.fps))
        self.smoothing = smoothing          # 1.0 = no smoothing
        self.min_frames = min_frames        # frames of history needed before reporting
        self.max_missing_frames = max_missing_frames
        self._history = {}                  # track_id -> deque[(frame_idx, x, y)]
        self._smoothed = {}                 # track_id -> last smoothed km/h
        self._last_seen = {}                # track_id -> last frame_idx

    def update(self, track_id, point, frame_idx):
        """Add the latest (x, y) pixel position of a track; return speed in km/h."""
        hist = self._history.setdefault(track_id, deque())
        hist.append((frame_idx, point[0], point[1]))
        self._last_seen[track_id] = frame_idx

        while hist and frame_idx - hist[0][0] > self.window_frames:
            hist.popleft()

        f0, x0, y0 = hist[0]
        frames = frame_idx - f0
        if frames < self.min_frames:
            return self._smoothed.get(track_id, 0.0)

        metres = math.hypot(point[0] - x0, point[1] - y0) / self.pixels_per_meter
        kmph = (metres / (frames / self.fps)) * 3.6

        prev = self._smoothed.get(track_id)
        speed = kmph if prev is None else self.smoothing * kmph + (1 - self.smoothing) * prev
        self._smoothed[track_id] = speed
        return speed

    def cleanup(self, frame_idx):
        """Forget tracks that have not been seen for a while (keeps memory bounded)."""
        stale = [t for t, last in self._last_seen.items()
                 if frame_idx - last > self.max_missing_frames]
        for t in stale:
            self._history.pop(t, None)
            self._smoothed.pop(t, None)
            self._last_seen.pop(t, None)

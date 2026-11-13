"""Audio alert: plays your .wav file without freezing the video loop."""
import shutil
import subprocess
import sys
import time
from pathlib import Path


class AudioAlert:
    def __init__(self, wav_path, cooldown=2.0, enabled=True):
        self.wav_path = Path(wav_path) if wav_path else None
        self.cooldown = cooldown      # minimum seconds between alerts
        self.enabled = enabled
        self._last = float('-inf')   # no alert played yet
        if self.enabled and (self.wav_path is None or not self.wav_path.is_file()):
            print(f"[alerts] sound file not found ({wav_path}); falling back to a system beep.")
            self.wav_path = None

    def trigger(self):
        if not self.enabled:
            return False
        now = time.monotonic()
        if now - self._last < self.cooldown:
            return False
        self._last = now
        self._play()
        return True

    def _play(self):
        path = str(self.wav_path) if self.wav_path else None
        try:
            if sys.platform.startswith("win"):
                import winsound
                if path:
                    winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                else:
                    winsound.MessageBeep()          # asynchronous, unlike winsound.Beep
                return
            if path:
                player = None
                if sys.platform == "darwin":
                    player = ["afplay", path]
                else:
                    for exe, args in (("paplay", [path]), ("aplay", ["-q", path]),
                                      ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", path])):
                        if shutil.which(exe):
                            player = [exe] + args
                            break
                if player and shutil.which(player[0]):
                    subprocess.Popen(player, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return
            print("\a", end="", flush=True)          # terminal bell as a last resort
        except Exception as exc:                      # never let audio crash the pipeline
            print(f"[alerts] could not play sound: {exc}")

"""YOLOv8 detection + tracking wrapper."""
from dataclasses import dataclass

# COCO names. Note: COCO calls motorbikes "motorcycle" (the word "motorbike" never matches).
DEFAULT_CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle", "cow")
ALIASES = {"motorbike": "motorcycle"}


@dataclass
class Detection:
    track_id: int
    label: str
    box: tuple          # (x1, y1, x2, y2) in pixels
    conf: float


class YoloTracker:
    def __init__(self, weights="models/yolov8n.pt", classes=DEFAULT_CLASSES,
                 conf=0.3, imgsz=640, tracker="botsort.yaml"):
        from ultralytics import YOLO                 # imported here so the rest works without it
        self.model = YOLO(weights)
        self.classes = {ALIASES.get(c, c) for c in classes}
        unknown = self.classes - set(self.model.names.values())
        if unknown:
            raise ValueError(f"Unknown class name(s): {sorted(unknown)}")
        self.conf, self.imgsz, self.tracker = conf, imgsz, tracker

    def __call__(self, frame_bgr):
        # OpenCV frames are BGR and Ultralytics expects BGR, so no colour conversion here.
        res = self.model.track(frame_bgr, persist=True, conf=self.conf, imgsz=self.imgsz,
                               tracker=self.tracker, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:        # no confirmed tracks in this frame
            return []
        ids = boxes.id.int().cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        out = []
        for tid, c, cf, b in zip(ids, clss, confs, xyxy):
            name = self.model.names[c]
            if name in self.classes:
                out.append(Detection(tid, name, tuple(int(v) for v in b), cf))
        return out

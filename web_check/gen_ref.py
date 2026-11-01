"""Reference detections: Ultralytics on the same ONNX file (CPU), conf 0.25, NMS IoU 0.45 -> web_check/ref.json."""
import json
from pathlib import Path
from ultralytics import YOLO

root = Path(__file__).resolve().parents[1]
r = YOLO(str(root / "docs/yolo11n.onnx"), task="detect").predict(
    str(root / "docs/sample.jpg"), imgsz=640, conf=0.25, iou=0.45, device="cpu", verbose=False)[0]
out = [{"box": b.xyxy[0].tolist(), "score": float(b.conf), "cls": int(b.cls)} for b in r.boxes]
Path(__file__).with_name("ref.json").write_text(json.dumps(out))
print(len(out), "reference detections")

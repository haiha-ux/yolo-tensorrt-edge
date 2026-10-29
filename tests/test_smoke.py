"""Each backend loads and returns detections on one COCO image (needs prepare_data.py + export.py first)."""
from pathlib import Path

import pytest
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
IMG = sorted((ROOT / "data/eval/images").glob("*.jpg"))[0]
# ONNX goes last: CPU mode blanks CUDA_VISIBLE_DEVICES for the rest of the process
CASES = [("weights/yolo11n.pt", 0)] + [(f"engines/yolo11n_{p}.engine", 0) for p in ("fp32", "fp16", "int8", "int8fp16")] \
    + [("engines/yolo11n.onnx", "cpu")]


@pytest.mark.parametrize("path,dev", CASES)
def test_returns_detections(path, dev):
    r = YOLO(ROOT / path, task="detect").predict(IMG, imgsz=640, device=dev, verbose=False)[0]
    assert len(r.boxes) > 0

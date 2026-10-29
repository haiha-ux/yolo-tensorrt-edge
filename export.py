"""PyTorch -> ONNX (opset 17, static 640) -> TensorRT engines FP32 / FP16 / INT8 / INT8+FP16.

INT8 uses TensorRT's MinMax calibrator (Ultralytics' GPU default) fed with the 200
calibration images from prepare_data.py (disjoint from the 500 eval images).
"""
import json, shutil, subprocess, sys, time

import cv2
import numpy as np
import tensorrt as trt
import torch
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

ROOT = Path(__file__).parent
W = ROOT / "weights" / "yolo11n.pt"
OUT = ROOT / "engines"


def to_onnx():
    onnx = YOLO(W).export(format="onnx", imgsz=640, opset=17, dynamic=False, simplify=False, batch=1)
    shutil.copy(onnx, OUT / "yolo11n.onnx")


def to_engine(prec):
    kw = dict(format="engine", imgsz=640, batch=1, workspace=4, dynamic=False,
              half=prec == "fp16", int8=prec == "int8")
    if prec == "int8":
        kw["data"] = str(ROOT / "data" / "calib.yaml")
    shutil.move(YOLO(W).export(**kw), OUT / f"yolo11n_{prec}.engine")


class MinMaxCalib(trt.IInt8MinMaxCalibrator):
    """Feeds the 200 calib images (letterboxed to 640, RGB, /255) one at a time."""

    def __init__(self, cache):
        super().__init__()
        self.files = sorted((ROOT / "data/calib/images").glob("*.jpg"))
        self.i, self.cache = 0, cache
        self.buf = torch.empty(1, 3, 640, 640, device="cuda")
        self.lb = LetterBox((640, 640), auto=False)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.i >= len(self.files):
            return None
        im = self.lb(image=cv2.imread(str(self.files[self.i])))[..., ::-1]  # BGR -> RGB
        self.buf.copy_(torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))) / 255.0)
        self.i += 1
        return [int(self.buf.data_ptr())]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, c):
        Path(self.cache).write_bytes(c)


def to_int8_fp16():
    """Own builder: INT8 + FP16 flags, so layers without an INT8 kernel fall back to FP16, not FP32."""
    logger = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(logger)
    net = b.create_network()
    assert trt.OnnxParser(net, logger).parse((OUT / "yolo11n.onnx").read_bytes()), "ONNX parse failed"
    cfg = b.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.set_flag(trt.BuilderFlag.FP16)
    # TRT 10.16 / Blackwell: levels >= 2 (default is 3) fail in Myelin ("no implementation for node
    # Quantize..output0_castOut") for this INT8+FP16 graph; level 1 builds. Recorded in README.
    cfg.builder_optimization_level = 1
    cfg.int8_calibrator = MinMaxCalib(OUT / "calib_int8fp16.cache")
    plan = b.build_serialized_network(net, cfg)
    assert plan, "engine build failed"
    ref = YOLO(W)
    meta = json.dumps({"batch": 1, "imgsz": [640, 640], "names": ref.names, "task": "detect", "stride": 32,
                       "args": {"batch": 1, "half": False, "dynamic": False, "int8": True}}).encode()
    (OUT / "yolo11n_int8fp16.engine").write_bytes(len(meta).to_bytes(4, "little") + meta + bytes(plan))


def main():
    OUT.mkdir(exist_ok=True)
    log = {}
    # each stage in its own process: Ultralytics' CPU ONNX export blanks CUDA_VISIBLE_DEVICES
    # for the rest of that process, which would hide the GPU from the TensorRT build.
    for stage in ("onnx", "fp32", "fp16", "int8", "int8fp16"):
        t = time.time()
        r = subprocess.run([sys.executable, __file__, stage])
        log[f"{stage}_s"] = round(time.time() - t, 1)
        if r.returncode:
            log[f"{stage}_error"] = f"exit {r.returncode}"
            print(f"[export] {stage} FAILED", file=sys.stderr)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "export_log.json").write_text(json.dumps(log, indent=2))
    print(log)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "onnx":
        to_onnx()
    elif sys.argv[1] == "int8fp16":
        to_int8_fp16()
    else:
        to_engine(sys.argv[1])

"""mAP50-95 / mAP50 on the 500-image COCO val2017 subset (Ultralytics val, conf 0.001, NMS iou 0.7).

Each model is validated in its own process (Ultralytics' CPU mode blanks CUDA_VISIBLE_DEVICES for the
rest of the process). Writes results/eval.json.
"""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent
E = ROOT / "engines"
MODELS = {
    "PyTorch eager|FP32": (ROOT / "weights/yolo11n.pt", "0"),
    "ONNX Runtime|FP32": (E / "yolo11n.onnx", "cpu"),  # CPU EP: this machine's ORT has no CUDA EP
    "TensorRT|FP32": (E / "yolo11n_fp32.engine", "0"),
    "TensorRT|FP16": (E / "yolo11n_fp16.engine", "0"),
    "TensorRT|INT8": (E / "yolo11n_int8.engine", "0"),
    "TensorRT|INT8+FP16": (E / "yolo11n_int8fp16.engine", "0"),
}


def val_one(key):
    from ultralytics import YOLO
    path, dev = MODELS[key]
    r = YOLO(path, task="detect").val(data=str(ROOT / "data/eval.yaml"), imgsz=640, batch=1, conf=0.001,
                                      iou=0.7, device=dev, plots=False, verbose=False)
    print("RESULT", json.dumps({"map50_95": round(float(r.box.map), 4), "map50": round(float(r.box.map50), 4)}))


def main():
    out = {}
    for key in MODELS:
        p = subprocess.run([sys.executable, __file__, key], capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        out[key] = json.loads(line[0][7:]) if line else {"error": p.stderr[-500:]}
        print(key, out[key])
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    val_one(sys.argv[1]) if len(sys.argv) > 1 else main()

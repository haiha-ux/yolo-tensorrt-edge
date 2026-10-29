"""Latency / FPS benchmark, batch 1, 640x640. Writes results/bench.json.

Measures the model forward pass only (random FP32 input already on GPU; no letterbox, no NMS)
through Ultralytics' AutoBackend, so every backend goes through the same harness.
Timing: 100 warm-up runs, then 1000 timed runs (x3 passes, best p50 kept), each followed by torch.cuda.synchronize().
"""
import json, re, statistics, subprocess, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from ultralytics.nn.autobackend import AutoBackend

ROOT = Path(__file__).parent
E = ROOT / "engines"
WARM, RUNS, PASSES = 100, 1000, 3
MODELS = {  # name -> (path, precision label)
    "PyTorch eager": (ROOT / "weights/yolo11n.pt", "FP32"),
    "ONNX Runtime": (E / "yolo11n.onnx", "FP32"),
    "TensorRT": [(E / "yolo11n_fp32.engine", "FP32"), (E / "yolo11n_fp16.engine", "FP16"),
                 (E / "yolo11n_int8.engine", "INT8"),
                 (E / "yolo11n_int8fp16.engine", "INT8+FP16")],
}


def time_model(path, device):
    m = AutoBackend(path, device=device, fp16=False, verbose=False)
    x = torch.rand(1, 3, 640, 640, device=device)
    cuda = device.type == "cuda"
    ts = []
    with torch.inference_mode():
        for i in range(WARM + RUNS):
            if cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            m(x)
            if cuda:
                torch.cuda.synchronize()
            if i >= WARM:
                ts.append((time.perf_counter() - t) * 1e3)
    p50, p95 = (float(v) for v in np.percentile(ts, [50, 95]))
    mean = statistics.fmean(ts)
    return dict(mean_ms=round(mean, 3), p50_ms=round(p50, 3), p95_ms=round(p95, 3),
                fps=round(1000 / mean, 1), size_mb=round(Path(path).stat().st_size / 2**20, 1))


def trtexec_gpu_ms(engine):
    """Pure GPU compute time (median) as reported by trtexec -- no Python, no host copies."""
    raw = Path(engine).read_bytes()  # Ultralytics prepends [4-byte len][JSON metadata]; trtexec wants the bare plan
    plan = Path(engine).with_suffix(".plan")
    plan.write_bytes(raw[4 + int.from_bytes(raw[:4], "little"):])
    r = subprocess.run(["/usr/bin/trtexec", f"--loadEngine={plan}", "--warmUp=500", "--iterations=1000",
                        "--noDataTransfers", "--useSpinWait"], capture_output=True, text=True)
    m = re.search(r"GPU Compute Time:.*?median = ([\d.]+) ms", r.stdout)
    return float(m.group(1)) if m else None


def main():
    dev = torch.device("cuda:0")
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "ort_providers": ort.get_available_providers(), "runs": RUNS, "warmup": WARM, "results": []}
    ort_dev = dev if "CUDAExecutionProvider" in out["ort_providers"] else torch.device("cpu")
    for name, spec in MODELS.items():
        for path, prec in (spec if isinstance(spec, list) else [spec]):
            # The GPU is shared with unrelated jobs on this box, so take 3 passes and keep the best p50.
            passes = [time_model(path, ort_dev if name == "ONNX Runtime" else dev) for _ in range(PASSES)]
            r = dict(min(passes, key=lambda p: p["p50_ms"]), pass_p50_ms=[p["p50_ms"] for p in passes])
            if name == "TensorRT":
                r["trtexec_gpu_median_ms"] = trtexec_gpu_ms(path)
            r.update(backend=name, precision=prec,
                     device="cpu (CPUExecutionProvider)" if name == "ONNX Runtime" and ort_dev.type == "cpu" else "cuda")
            out["results"].append(r)
            print(r)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/bench.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

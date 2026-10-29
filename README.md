# yolo-tensorrt-edge

Take a YOLO model from PyTorch to ONNX to TensorRT (FP32, FP16, INT8), then measure latency, throughput and COCO accuracy with the same scripts for every backend, so the speed/accuracy trade-off is a number and not a claim. Everything here uses public assets only: Ultralytics YOLO11n official weights and a 500-image slice of COCO val2017.

![latency vs mAP](results/latency_vs_map.png)

## Results

YOLO11n, 640x640, batch 1, NVIDIA GB10 (Blackwell, aarch64 **desktop-class** GPU, CUDA 13, TensorRT 10.16, torch 2.11, ONNX Runtime 1.22). Raw data: `results/bench.json`, `results/eval.json`, `results/export_log.json`.

| Backend | Precision | mAP50-95 | mAP50 | p50 ms | p95 ms | FPS (batch 1) | File size MB |
|---|---|---|---|---|---|---|---|
| PyTorch eager | FP32 | 0.418 | 0.574 | 2.48 | 2.50 | 402 | 5.4 |
| ONNX Runtime (CPU) | FP32 | 0.424 | 0.577 | 28.82 | 43.28 | 32 | 10.2 |
| TensorRT | FP32 | 0.424 | 0.577 | 1.65 | 1.66 | 605 | 12.8 |
| TensorRT | FP16 | 0.424 | 0.577 | 0.85 | 0.86 | 1175 | 7.0 |
| TensorRT | INT8 (Ultralytics build) | 0.404 | 0.563 | 1.32 | 1.33 | 755 | 16.3 |
| TensorRT | INT8+FP16 (own builder, opt. level 1) | 0.404 | 0.562 | 1.98 | 1.99 | 505 | 52.5 |

Take-aways, plainly:
- **FP16 is the winner here**: about 2x faster than TensorRT FP32 and ~3x faster than PyTorch eager, with no measurable accuracy loss (0.424 vs 0.424).
- **INT8 lost accuracy (-2.0 mAP50-95 points) and was slower than FP16** on this GPU for this tiny model. Likely reasons: YOLO11n is small and launch/memory-bound rather than compute-bound, quantize/dequantize nodes add overhead, and the INT8 engine file is bigger than FP16. INT8 usually pays off on larger models and on hardware with dedicated INT8 paths (Jetson DLA/Orin tensor cores); this table does not show that, and I did not try to tune it (per-layer precision, QAT, more calibration images).
- INT8 calibration used 200 images (TensorRT recommends 300+) and a MinMax calibrator; calibration images are disjoint from eval images.
- Building INT8+FP16 with my own builder at TensorRT's default optimization level failed on this setup (`Myelin ... Could not find any implementation for node {ForeignNode[Quantize ...]}`); it only builds at `builder_optimization_level <= 1`, which is why that row is slower and larger. Treat it as a reproducible failure report, not a tuned result.
- ONNX Runtime ran on **CPU**: the ONNX Runtime wheel on this machine exposes no CUDA provider. Its row is a CPU baseline, not a GPU comparison.
- PyTorch's mAP is ~0.6 pt lower than the static-shape ONNX/TensorRT runs because Ultralytics validates `.pt` with rectangular (minimal-padding) batches while the exported models use fixed square 640x640 letterboxing. Same weights.

## How the numbers were measured

- **Accuracy** (`eval.py`): Ultralytics `val`, conf 0.001, NMS IoU 0.7, on the first 500 images (sorted by filename, with labels) of COCO val2017. 500 images is a subset: mAP is comparable across rows here, not to published full-val numbers.
- **Latency** (`bench.py`): model forward pass only, random FP32 input already on the GPU, no letterbox/NMS, `cuda.synchronize` around each call. 100 warm-up + 1000 timed runs, 3 passes, best-p50 pass reported (the GPU is a shared machine; per-pass p50s are in `bench.json`). FPS = 1000 / mean latency, i.e. sequential batch-1 throughput. Batched throughput was not measured (engines are static batch 1).
- `bench.json` also holds `trtexec` GPU-compute medians for the engines (kernel time only, a bit different from the end-to-end numbers above).
- **Not Jetson**: this is a GB10 desktop GPU. Jetson numbers will differ, in absolute value and likely in the FP16/INT8 ratio.

## Reproduce

```bash
pip install -r requirements.txt            # CUDA-enabled torch + TensorRT 10.x required
curl -L -o weights/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt   # mkdir weights first
python prepare_data.py     # downloads COCO val2017 (~800 MB) + labels, builds 500 eval / 200 calib subsets
python export.py           # ONNX + TensorRT FP32/FP16/INT8/INT8+FP16 into engines/ (~2 min)
python bench.py            # -> results/bench.json   (~4 min)
python eval.py             # -> results/eval.json    (~1 min)
python plot.py             # -> results/results.md, results/latency_vs_map.png
pytest -q tests            # each backend loads and detects on one image
```

Whole pipeline after the data download took about 6.5 minutes on this machine. Weights, engines and data are git-ignored.

## What I would do on a Jetson Orin / Qualcomm NPU (approach, not measured here)

- **Jetson Orin**: export ONNX on the host, build the TensorRT engine **on the device** (engines are not portable across GPU architectures/TRT versions), lock clocks with `nvpmodel`/`jetson_clocks` before benchmarking, and measure with `trtexec` plus an end-to-end pipeline (camera decode, letterbox, NMS) because pre/post-processing often dominates on small models. Try FP16 first, then INT8 with 500+ representative calibration frames from the real deployment scene, and check whether layers can run on the DLA with GPU fallback. Verify mAP on the customer's own labelled data, not only COCO.
- **Qualcomm NPU (e.g. Hexagon via QNN / AI Hub)**: export a static-shape ONNX, quantize to INT8/W8A8 with a calibration set from the target domain, compile with the Qualcomm toolchain, and compare on-device accuracy and latency to the FP32 reference layer by layer when accuracy drops. Detection heads (box decode, NMS) are typically kept off the NPU or simplified.
- In both cases the same workflow as this repo applies: one script per stage, an identical eval harness across precisions, and raw numbers saved next to the code.

## License

MIT. Author: Ha Phan Thanh Hai.

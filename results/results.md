| Backend | Precision | mAP50-95 | mAP50 | p50 ms | p95 ms | FPS (batch 1) | File size MB |
|---|---|---|---|---|---|---|---|
| PyTorch eager | FP32 | 0.418 | 0.574 | 2.48 | 2.50 | 402 | 5.4 |
| ONNX Runtime (CPU) | FP32 | 0.424 | 0.577 | 28.82 | 43.28 | 32 | 10.2 |
| TensorRT | FP32 | 0.424 | 0.577 | 1.65 | 1.66 | 605 | 12.8 |
| TensorRT | FP16 | 0.424 | 0.577 | 0.85 | 0.86 | 1175 | 7.0 |
| TensorRT | INT8 | 0.404 | 0.563 | 1.32 | 1.33 | 755 | 16.3 |
| TensorRT | INT8+FP16 | 0.404 | 0.562 | 1.98 | 1.99 | 505 | 52.5 |

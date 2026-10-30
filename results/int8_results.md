| Model | Batch | Precision | mAP50-95 | p50 ms | p95 ms | img/s | vs FP16 default | vs FP16 same opt level | trtexec GPU ms | Act. mem MB | Engine MB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11n | 1 | FP16 | 0.424 | 0.82 | 0.84 | 1203 | 1.00x | 1.00x | 0.89 | 16 | 7.0 |
| YOLO11n | 1 | FP16, opt level 1 (control) | 0.424 | 0.86 | 0.88 | 1151 | 0.95x | 1.00x | 0.92 | 16 | 6.2 |
| YOLO11n | 1 | INT8 entropy (FP32 fallback) | 0.300 | 1.31 | 3.03 | 655 | 0.63x | 0.63x | 1.38 | 26 | 18.5 |
| YOLO11n | 1 | INT8 minmax (FP32 fallback) | 0.406 | 1.31 | 1.32 | 759 | 0.63x | 0.63x | 1.32 | 26 | 18.6 |
| YOLO11n | 1 | INT8 + FP16 fallback, opt level 1 | 0.300 | 1.98 | 2.06 | 503 | 0.42x | 0.44x | 2.04 | 63 | 52.5 |
| YOLO11n | 1 | INT8, FP16 detection head, opt level 1 | 0.371 | 1.95 | 1.97 | 511 | 0.42x | 0.44x | 1.96 | 63 | 50.6 |
| YOLO11n | 8 | FP16 | 0.424 | 5.13 | 6.81 | 1425 | 1.00x | 1.00x | 4.97 | 125 | 6.3 |
| YOLO11n | 8 | FP16, opt level 1 (control) | 0.424 | 5.17 | 5.26 | 1540 | 0.99x | 1.00x | 7.34 | 106 | 12.7 |
| YOLO11n | 8 | INT8 entropy (FP32 fallback) | 0.300 | 7.71 | 7.82 | 1034 | 0.67x | 0.67x | 7.68 | 105 | 50.0 |
| YOLO11n | 8 | INT8 + FP16 fallback, opt level 1 | 0.300 | 13.76 | 14.62 | 576 | 0.37x | 0.38x | 13.98 | 505 | 387.7 |
| YOLO11n | 8 | INT8, FP16 detection head, opt level 1 | 0.371 | 12.87 | 13.71 | 612 | 0.40x | 0.40x | 14.06 | 505 | 370.5 |
| YOLO11m | 1 | FP16 | 0.555 | 2.64 | 2.97 | 374 | 1.00x | 1.00x | 2.62 | 55 | 40.6 |
| YOLO11m | 1 | FP16, opt level 1 (control) | 0.555 | 2.71 | 2.73 | 367 | 0.97x | 1.00x | 2.71 | 55 | 40.7 |
| YOLO11m | 1 | INT8 entropy (FP32 fallback) | 0.490 | 2.99 | 3.24 | 331 | 0.88x | 0.88x | 2.99 | 45 | 26.6 |
| YOLO11m | 1 | INT8 minmax (FP32 fallback) | 0.537 | 2.94 | 2.98 | 338 | 0.90x | 0.90x | 2.95 | 45 | 26.6 |
| YOLO11m | 1 | INT8 + FP16 fallback, opt level 1 | 0.491 | 3.38 | 3.42 | 295 | 0.78x | 0.80x | 3.87 | 76 | 69.0 |
| YOLO11m | 1 | INT8, FP16 detection head, opt level 1 | 0.497 | 3.33 | 3.64 | 297 | 0.79x | 0.81x | 3.32 | 76 | 68.1 |
| YOLO11m | 8 | FP16 | 0.555 | 21.94 | 25.37 | 359 | 1.00x | 1.00x | 22.10 | 441 | 46.2 |
| YOLO11m | 8 | FP16, opt level 1 (control) | 0.555 | 22.42 | 23.42 | 357 | 0.98x | 1.00x | 23.10 | 441 | 46.1 |
| YOLO11m | 8 | INT8 entropy (FP32 fallback) | 0.490 | 36.04 | 49.63 | 203 | 0.61x | 0.61x | 36.97 | 608 | 384.2 |
| YOLO11m | 8 | INT8 + FP16 fallback, opt level 1 | 0.491 | 33.51 | 37.12 | 235 | 0.65x | 0.67x | 34.35 | 608 | 400.6 |
| YOLO11m | 8 | INT8, FP16 detection head, opt level 1 | 0.497 | 31.84 | 37.01 | 244 | 0.69x | 0.70x | 31.31 | 608 | 385.6 |
| YOLO11l | 1 | FP16 | 0.569 | 3.41 | 3.82 | 287 | 1.00x | 1.00x | 3.39 | 64 | 50.8 |
| YOLO11l | 1 | FP16, opt level 1 (control) | 0.569 | 3.46 | 3.62 | 286 | 0.98x | 1.00x | 4.37 | 64 | 50.8 |
| YOLO11l | 1 | INT8 entropy (FP32 fallback) | 0.510 | 3.71 | 4.37 | 264 | 0.92x | 0.92x | 4.39 | 49 | 32.5 |
| YOLO11l | 1 | INT8 minmax (FP32 fallback) | 0.553 | 3.78 | 3.82 | 263 | 0.90x | 0.90x | 3.80 | 49 | 32.5 |
| YOLO11l | 1 | INT8 + FP16 fallback, opt level 1 | 0.512 | 3.88 | 4.22 | 254 | 0.88x | 0.89x | 4.22 | 80 | 74.8 |
| YOLO11l | 1 | INT8, FP16 detection head, opt level 1 | 0.521 | 3.79 | 3.92 | 262 | 0.90x | 0.91x | 3.81 | 80 | 74.0 |
| YOLO11l | 8 | FP16 | 0.569 | 27.81 | 38.63 | 272 | 1.00x | 1.00x | 30.41 | 512 | 56.3 |
| YOLO11l | 8 | FP16, opt level 1 (control) | 0.569 | 28.30 | 39.81 | 268 | 0.98x | 1.00x | 29.05 | 512 | 56.2 |
| YOLO11l | 8 | INT8 entropy (FP32 fallback) | 0.510 | 34.97 | 40.49 | 222 | 0.80x | 0.80x | 34.68 | 394 | 68.4 |
| YOLO11l | 8 | INT8 + FP16 fallback, opt level 1 | 0.512 | 37.31 | 41.27 | 210 | 0.75x | 0.76x | 37.71 | 644 | 406.5 |
| YOLO11l | 8 | INT8, FP16 detection head, opt level 1 | 0.521 | 36.58 | 49.76 | 205 | 0.76x | 0.77x | 36.48 | 644 | 391.3 |

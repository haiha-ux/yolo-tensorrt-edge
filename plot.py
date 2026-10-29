"""results/bench.json + results/eval.json -> results/latency_vs_map.png and results/results.md (README table)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).parent / "results"
bench = json.loads((R / "bench.json").read_text())["results"]
ev = json.loads((R / "eval.json").read_text())
rows = [dict(b, **ev[f"{b['backend']}|{b['precision']}"]) for b in bench]

md = ["| Backend | Precision | mAP50-95 | mAP50 | p50 ms | p95 ms | FPS (batch 1) | File size MB |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    name = r["backend"] + (" (CPU)" if r["device"].startswith("cpu") else "")
    md.append(f"| {name} | {r['precision']} | {r['map50_95']:.3f} | {r['map50']:.3f} | {r['p50_ms']:.2f} | "
              f"{r['p95_ms']:.2f} | {r['fps']:.0f} | {r['size_mb']:.1f} |")
(R / "results.md").write_text("\n".join(md) + "\n")

fig, ax = plt.subplots(figsize=(7, 4.5))
for r in rows:
    ax.scatter(r["p50_ms"], r["map50_95"], s=70)
    ax.annotate(f"{r['backend'].replace('PyTorch eager','PyTorch').replace('ONNX Runtime','ORT (CPU)')} {r['precision']}", (r["p50_ms"], r["map50_95"]),
                textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("p50 latency, ms (batch 1, log scale; lower = faster)")
ax.set_ylabel("mAP50-95 (500 COCO val2017 images)")
ax.set_title("YOLO11n 640x640 on NVIDIA GB10: latency vs accuracy")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(R / "latency_vs_map.png", dpi=140)

"""When does INT8 pay off? YOLO11 n/m/l x batch 1/8 x {FP16, INT8(+FP16 fallback), INT8 with FP16 detection head}.

Stages (each in its own process, resumable):  python3 int8_study.py build|bench|eval|report
 build : static ONNX per batch -> TensorRT engine; INT8 uses the entropy calibrator (IInt8EntropyCalibrator2) on
         512 COCO train2017 images (prepare_calib_train.py); the calibration table is cached in results/calib/.
 bench : same timing harness as bench.py (AutoBackend, 100 warm-up + 1000 runs x3 passes, best p50), + trtexec GPU
         median and engine activation memory.
 eval  : same Ultralytics val as eval.py (500 val2017 images, conf 0.001, iou 0.7), on the batch-1 engines.
Writes results/int8_{build,bench,eval}.json, results/int8_results.csv/.md and results/int8_vs_fp16.png.
"""
import csv, json, re, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
E, R = ROOT / "engines", ROOT / "results"
MODELS, BATCHES = ("yolo11n", "yolo11m", "yolo11l"), (1, 8)
# variant -> (INT8 flag, FP16 flag, builder_optimization_level or None=TensorRT default 3, FP16 detection head, label).
# INT8+FP16 only builds at level 1 on this TRT 10.16/Blackwell setup (Myelin failure at level>=2, see README), so
# "fp16_l1" is the like-for-like FP16 control for those rows.
CFG = {"fp16": (0, 1, None, 0, "FP16"),
       "fp16_l1": (0, 1, 1, 0, "FP16, opt level 1 (control)"),
       "int8": (1, 0, None, 0, "INT8 entropy (FP32 fallback)"),
       "int8mm": (1, 0, None, 0, "INT8 minmax (FP32 fallback)"),
       "int8fp16": (1, 1, 1, 0, "INT8 + FP16 fallback, opt level 1"),
       "int8head": (1, 1, 1, 1, "INT8, FP16 detection head, opt level 1")}
VARIANTS = tuple(CFG)
LABEL = {k: v[4] for k, v in CFG.items()}


def eng(m, v, b):
    return E / f"{m}_{v}_b{b}.engine"


def jload(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------- build
def make_calib(cache, kind):
    import cv2, numpy as np, tensorrt as trt, torch
    from ultralytics.data.augment import LetterBox

    class Calib(trt.IInt8EntropyCalibrator2 if kind == "entropy" else trt.IInt8MinMaxCalibrator):
        def __init__(self):
            super().__init__()
            self.files = sorted((ROOT / "data/calib_train/images").glob("*.jpg"))
            assert len(self.files) >= 500, "run prepare_calib_train.py first"
            self.i, self.lb = 0, LetterBox((640, 640), auto=False)
            self.buf = torch.empty(1, 3, 640, 640, device="cuda")

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.i >= len(self.files):
                return None
            im = self.lb(image=cv2.imread(str(self.files[self.i])))[..., ::-1]
            self.buf.copy_(torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))) / 255.0)
            self.i += 1
            return [int(self.buf.data_ptr())]

        def read_calibration_cache(self):
            return cache.read_bytes() if cache.exists() else None

        def write_calibration_cache(self, c):
            cache.write_bytes(c)

    return Calib()


def build_one(m, v, b):
    import tensorrt as trt
    from ultralytics import YOLO
    onnx = E / f"{m}_b{b}.onnx"
    if not onnx.exists():  # CPU export blanks CUDA_VISIBLE_DEVICES, so do it in a child process
        subprocess.run([sys.executable, __file__, "onnx", m, str(b)], check=True)
    kind = "minmax" if v == "int8mm" else "entropy"  # calibrator; all other INT8 variants use entropy
    cache = R / "calib" / f"{m}_{kind}.cache"
    cache.parent.mkdir(exist_ok=True)
    log = trt.Logger(trt.Logger.WARNING)
    bld = trt.Builder(log)
    net = bld.create_network()
    assert trt.OnnxParser(net, log).parse(onnx.read_bytes()), "ONNX parse failed"
    cfg = bld.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    i8, f16, level, head, _ = CFG[v]
    n_head = 0
    if i8:
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = make_calib(cache, kind)
    if f16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    if level:
        cfg.builder_optimization_level = level
    if head:  # detection head = layers under /model.23/ (Detect); keep them FP16
        cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        for i in range(net.num_layers):
            L = net.get_layer(i)
            if "/model.23/" in L.name and L.get_output(0).dtype == trt.float32 and L.type not in (
                    trt.LayerType.SHAPE, trt.LayerType.CONSTANT):
                L.precision = trt.float16
                for k in range(L.num_outputs):
                    L.set_output_type(k, trt.float16)
                n_head += 1
        assert n_head, "no head layers found"
    t = time.time()
    plan = bld.build_serialized_network(net, cfg)
    assert plan, "engine build failed"
    ref = YOLO(ROOT / "weights" / f"{m}.pt")
    meta = json.dumps({"batch": b, "imgsz": [640, 640], "names": ref.names, "task": "detect", "stride": 32,
                       "args": {"batch": b, "half": bool(f16), "dynamic": False, "int8": bool(i8)}}).encode()
    eng(m, v, b).write_bytes(len(meta).to_bytes(4, "little") + meta + bytes(plan))
    info = dict(build_s=round(time.time() - t, 1), opt_level=level or "default(3)", head_layers_fp16=n_head,
                calib_images=len(list((ROOT / "data/calib_train/images").glob("*.jpg"))) if i8 else 0)
    (R / f"int8_build_{m}_{v}_b{b}.json").write_text(json.dumps(info))


def to_onnx(m, b):
    from ultralytics import YOLO
    onnx = YOLO(ROOT / "weights" / f"{m}.pt").export(format="onnx", imgsz=640, opset=17, dynamic=False,
                                                    simplify=False, batch=b)
    shutil.move(onnx, E / f"{m}_b{b}.onnx")


def stage_build():
    E.mkdir(exist_ok=True)
    for m in MODELS:
        for b in BATCHES:  # b=1 first so the calibration table exists
            for v in VARIANTS:
                if eng(m, v, b).exists() or (v == "int8mm" and b == 8):  # minmax: accuracy row at batch 1 only
                    continue
                r = subprocess.run([sys.executable, __file__, "build1", m, v, str(b)])
                print(m, v, b, "FAILED" if r.returncode else "ok", flush=True)
        for b in BATCHES:
            (E / f"{m}_b{b}.onnx").unlink(missing_ok=True)


# ---------------------------------------------------------------- bench
def trtexec(engine):
    raw = engine.read_bytes()
    plan = engine.with_suffix(".plan")
    plan.write_bytes(raw[4 + int.from_bytes(raw[:4], "little"):])
    r = subprocess.run(["/usr/bin/trtexec", f"--loadEngine={plan}", "--warmUp=500", "--iterations=500",
                        "--noDataTransfers", "--useSpinWait"], capture_output=True, text=True)
    plan.unlink()
    g = re.search(r"GPU Compute Time:.*?median = ([\d.]+) ms", r.stdout)
    return float(g.group(1)) if g else None


def act_mem_mb(engine):
    import tensorrt as trt
    raw = engine.read_bytes()
    plan = raw[4 + int.from_bytes(raw[:4], "little"):]
    e = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    return round(e.device_memory_size_v2 / 2**20, 1)


def bench_one(m, v, b):
    import torch
    from bench import time_model
    passes = [time_model(eng(m, v, b), torch.device("cuda:0"), b) for _ in range(3)]
    r = dict(min(passes, key=lambda p: p["p50_ms"]), pass_p50_ms=[p["p50_ms"] for p in passes])
    r.update(model=m, variant=v, batch=b, img_per_s=round(1000 * b / r["mean_ms"], 1),
             trtexec_gpu_median_ms=trtexec(eng(m, v, b)), act_mem_mb=act_mem_mb(eng(m, v, b)))
    print("RESULT", json.dumps(r))


def stage_bench():
    out = jload("int8_bench.json")
    for m in MODELS:
        for b in BATCHES:
            for v in VARIANTS:
                k = f"{m}|{v}|{b}"
                if k in out or not eng(m, v, b).exists():
                    continue
                p = subprocess.run([sys.executable, __file__, "bench1", m, v, str(b)], capture_output=True, text=True)
                line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
                out[k] = json.loads(line[0][7:]) if line else {"error": p.stderr[-400:]}
                print(k, out[k], flush=True)
                (R / "int8_bench.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------- eval
def eval_one(m, v):
    from ultralytics import YOLO
    r = YOLO(eng(m, v, 1), task="detect").val(data=str(ROOT / "data/eval.yaml"), imgsz=640, batch=1, conf=0.001,
                                              iou=0.7, device="0", plots=False, verbose=False)
    print("RESULT", json.dumps({"map50_95": round(float(r.box.map), 4), "map50": round(float(r.box.map50), 4)}))


def stage_eval():
    out = jload("int8_eval.json")
    for m in MODELS:
        for v in VARIANTS:
            k = f"{m}|{v}"
            if k in out or not eng(m, v, 1).exists():
                continue
            p = subprocess.run([sys.executable, __file__, "eval1", m, v], capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
            out[k] = json.loads(line[0][7:]) if line else {"error": p.stderr[-400:]}
            print(k, out[k], flush=True)
            (R / "int8_eval.json").write_text(json.dumps(out, indent=1))


# ---------------------------------------------------------------- report
def stage_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bench, ev, builds = jload("int8_bench.json"), jload("int8_eval.json"), jload("int8_build.json")
    for p in R.glob("int8_build_*.json"):  # per-build files -> one int8_build.json
        builds[p.stem[len("int8_build_"):]] = json.loads(p.read_text())
        p.unlink()
    (R / "int8_build.json").write_text(json.dumps(builds, indent=1))
    rows = []
    for m in MODELS:
        for b in BATCHES:
            for v in VARIANTS:
                r = bench.get(f"{m}|{v}|{b}")
                if not r or "error" in r:
                    continue
                bi = builds[f"{m}_{v}_b{b}"]
                e = ev.get(f"{m}|{v}", {})
                base = bench[f"{m}|fp16|{b}"]
                base1 = bench[f"{m}|fp16_l1|{b}"]
                rows.append(dict(model=m, batch=b, precision=LABEL[v], map50_95=e.get("map50_95"),
                                 p50_ms=r["p50_ms"], p95_ms=r["p95_ms"], img_per_s=r["img_per_s"],
                                 speedup_vs_fp16=round(base["p50_ms"] / r["p50_ms"], 2),
                                 speedup_vs_fp16_same_level=round((base1 if bi["opt_level"] == 1 else base)["p50_ms"] / r["p50_ms"], 2),
                                 trtexec_gpu_median_ms=r["trtexec_gpu_median_ms"], act_mem_mb=r["act_mem_mb"],
                                 engine_mb=r["size_mb"], opt_level=bi["opt_level"]))
    with open(R / "int8_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    md = ["| Model | Batch | Precision | mAP50-95 | p50 ms | p95 ms | img/s | vs FP16 default | vs FP16 same opt level | trtexec GPU ms | Act. mem MB | Engine MB |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model'].replace('yolo11', 'YOLO11')} | {r['batch']} | {r['precision']} | {r['map50_95']:.3f} | "
                  f"{r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['img_per_s']:.0f} | {r['speedup_vs_fp16']:.2f}x | "
                  f"{r['speedup_vs_fp16_same_level']:.2f}x | "
                  f"{r['trtexec_gpu_median_ms']:.2f} | {r['act_mem_mb']:.0f} | {r['engine_mb']:.1f} |")
    (R / "int8_results.md").write_text("\n".join(md) + "\n")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, b in zip(axs, BATCHES):
        for i, v in enumerate(VARIANTS):
            xs = [j + (i - 2.5) * 0.13 for j, m in enumerate(MODELS)]
            ys = [next((r["speedup_vs_fp16"] for r in rows if r["model"] == m and r["batch"] == b
                        and r["precision"] == LABEL[v]), 0) for m in MODELS]
            ax.bar(xs, ys, 0.13, label=LABEL[v])
        ax.axhline(1, color="k", lw=.8)
        ax.set_xticks(range(len(MODELS)), [m.replace("yolo11", "YOLO11") for m in MODELS])
        ax.set_title(f"batch {b}")
        ax.set_ylim(0, 1.6)
        ax.grid(alpha=.3, axis="y")
    axs[0].set_ylabel("speed vs FP16 (p50 latency ratio; >1 = faster)")
    axs[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("INT8 vs FP16 on NVIDIA GB10 (shared desktop GPU), 640x640")
    fig.tight_layout()
    fig.savefig(R / "int8_vs_fp16.png", dpi=140)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "onnx":
        to_onnx(a[1], int(a[2]))
    elif a[0] == "build1":
        build_one(a[1], a[2], int(a[3]))
    elif a[0] == "bench1":
        bench_one(a[1], a[2], int(a[3]))
    elif a[0] == "eval1":
        eval_one(a[1], a[2])
    else:
        {"build": stage_build, "bench": stage_bench, "eval": stage_eval, "report": stage_report}[a[0]]()

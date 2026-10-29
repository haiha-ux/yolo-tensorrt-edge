"""Build eval (500 imgs) and calibration (200 imgs, disjoint) subsets from public COCO val2017.

Downloads val2017.zip + Ultralytics' coco2017labels.zip into data/ if missing.
"""
import os, urllib.request, zipfile
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
D = ROOT / "data"
URLS = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "labels.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip",
}
N_EVAL, N_CALIB = 500, 200
NAMES = YOLO(ROOT / "weights/yolo11n.pt").names  # COCO-80 class names


def main():
    D.mkdir(exist_ok=True)
    for name, url in URLS.items():
        if not (D / name).exists():
            print("downloading", url)
            urllib.request.urlretrieve(url, D / name)
    if not (D / "val2017").exists():
        zipfile.ZipFile(D / "val2017.zip").extractall(D)
    if not (D / "coco/labels/val2017").exists():
        z = zipfile.ZipFile(D / "labels.zip")
        z.extractall(D, [n for n in z.namelist() if n.startswith("coco/labels/val2017/")])

    imgs = sorted(p.name for p in (D / "val2017").glob("*.jpg"))
    # only images that have a label file (COCO val has ~48 images with no objects; skipped)
    imgs = [i for i in imgs if (D / "coco/labels/val2017" / (Path(i).stem + ".txt")).exists()]
    subsets = {"eval": imgs[:N_EVAL], "calib": imgs[N_EVAL:N_EVAL + N_CALIB]}
    assert not set(subsets["eval"]) & set(subsets["calib"])
    for name, files in subsets.items():
        for sub in ("images", "labels"):
            (D / name / sub).mkdir(parents=True, exist_ok=True)
        for f in files:
            for sub, src in (("images", D / "val2017" / f),
                             ("labels", D / "coco/labels/val2017" / (Path(f).stem + ".txt"))):
                dst = D / name / sub / src.name
                if not dst.exists():
                    os.symlink(src.resolve(), dst)
        (D / f"{name}.yaml").write_text(
            f"path: {(D / name).resolve()}\ntrain: images\nval: images\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in NAMES.items()))
    print({k: len(v) for k, v in subsets.items()})


if __name__ == "__main__":
    main()

"""Download 512 COCO *train2017* images (seeded random sample, only images that have labels) for INT8 calibration.

Disjoint from the val2017 eval subset by construction (different split). ~80 MB, sequential, throttled.
"""
import random, time, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "data/calib_train/images"
N = 512


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ROOT / "data/labels.zip")
    ids = sorted(Path(n).stem for n in z.namelist() if "labels/train2017/" in n and n.endswith(".txt")
                 and z.getinfo(n).file_size > 0)
    for i in random.Random(0).sample(ids, N):
        dst = OUT / f"{i}.jpg"
        if not dst.exists():
            urllib.request.urlretrieve(f"http://images.cocodataset.org/train2017/{i}.jpg", dst)
            time.sleep(0.05)
    print(len(list(OUT.glob("*.jpg"))), "calibration images")


if __name__ == "__main__":
    main()

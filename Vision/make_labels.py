"""用現有的傳統辨識（find_pieces／stocks）幫俯視影像自動標註 O／X，輸出候選標註＋檢查用拼圖。
    python make_labels.py        # 寫 Vision/work/cands.json、Vision/work/sheet_*.jpg
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Calib"))
import find_pieces   # noqa: E402
import stocks        # noqa: E402

W = os.path.join(HERE, "work")
os.makedirs(W, exist_ok=True)
plan = json.load(open(os.path.join(ROOT, "Calib", "cell_plan.json"), encoding="utf-8"))


def dhash(im):
    g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (17, 16))
    return (g[:, 1:] > g[:, :-1]).flatten()


files = []
for d in ("Calib/out", "Arena/out", "Gemini2/out", "Calib", "Arena"):
    files += glob.glob(os.path.join(ROOT, d, "*.png")) + glob.glob(os.path.join(ROOT, d, "*.jpg"))
seen, items = [], []
for f in sorted(files):
    if os.path.getsize(f) < 60000:
        continue
    im = cv2.imread(f)
    if im is None or im.shape[:2] != (720, 1280):
        continue
    if "depth" in os.path.basename(f) or cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 1].mean() < 8:
        continue                                       # 深度圖／灰階圖不是俯視彩色畫面
    print(len(items), os.path.basename(f), flush=True)
    h = dhash(im)
    if any((h != s).sum() < 6 for s in seen):          # 幾乎一樣的畫面只留一張
        continue
    seen.append(h)
    c = []
    try:
        for p in find_pieces.pieces(im, plan):
            c.append([float(p["px"][0]), float(p["px"][1]), 53.0, p["kind"]])
    except Exception:
        pass
    try:
        for u, v, r in stocks.disc_peaks(im, min_r=30):
            if any(np.hypot(u - a[0], v - a[1]) < 55 for a in c):
                continue
            if not (36 <= r <= 75) or u < 40 or v < 40 or u > 1240 or v > 680:
                continue
            c.append([float(u), float(v), float(min(max(r, 50), 58)), stocks.kind_of(im, u, v, 52)])
    except Exception:
        pass
    items.append({"file": os.path.relpath(f, ROOT).replace("\\", "/"), "cands": c})
json.dump(items, open(os.path.join(W, "cands.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("影像 %d 張（去重後），候選 %d 個：" % (len(items), sum(len(i["cands"]) for i in items)),
      {k: sum(1 for i in items for c in i["cands"] if c[3] == k) for k in ("O", "X", "?")})
# 檢查用拼圖：每張縮小、畫上標註與編號
for s in range(0, len(items), 12):
    tiles = []
    for k, it in enumerate(items[s:s + 12]):
        im = cv2.imread(os.path.join(ROOT, it["file"]))
        for j, (u, v, r, kd) in enumerate(it["cands"]):
            col = {"O": (255, 120, 0), "X": (0, 0, 255), "?": (0, 255, 255)}[kd]
            cv2.circle(im, (int(u), int(v)), int(r), col, 4)
            cv2.putText(im, "%s%d" % (kd, j), (int(u) - 25, int(v) + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
        cv2.putText(im, "#%d" % (s + k), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
        tiles.append(cv2.resize(im, (640, 360)))
    while len(tiles) % 3:
        tiles.append(np.zeros((360, 640, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(os.path.join(W, "sheet_%02d.jpg" % (s // 12)), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])

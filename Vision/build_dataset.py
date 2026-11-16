"""把 work/cands.json（自動標註）＋人工修正 → YOLO 資料集（dataset/），含光線擴增。
    python build_dataset.py
修正表是 2026-11-15 逐張看裁切拼圖（work/crops_*.jpg）訂的；重跑 make_labels.py 後編號會變，要重新檢查。"""
import json
import os
import random
import shutil

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
random.seed(7); np.random.seed(7)

DROP_IMAGES = {10, 65, 73, 68}                       # 畫過記號的除錯圖、比例不同的圖
RELABEL = {(45, 4): "X", (49, 0): "O", (80, 0): "O"}
REMOVE = {(15, 4), (15, 5), (36, 3), (45, 5), (49, 1), (71, 2), (80, 1), (80, 3)}          # 不是棋子（LED、手、機構）
IGNORE = {(8, 4), (48, 2), (1, 8), (14, 3), (18, 3), (24, 3), (25, 3), (40, 0), (43, 0),
          (46, 1), (41, 6), (42, 7), (47, 6)}        # 是棋子但被遮住／出畫面：塗掉，不當正例也不當背景
CLS = {"O": 0, "X": 1}

items = json.load(open(os.path.join(HERE, "work", "cands.json"), encoding="utf-8"))
D = os.path.join(HERE, "dataset")
shutil.rmtree(D, ignore_errors=True)
for s in ("train", "val"):
    os.makedirs(os.path.join(D, "images", s)); os.makedirs(os.path.join(D, "labels", s))


def photometric(im):
    """模擬現場光線變化：亮度／對比／gamma、色溫、陰影漸層、反光亮斑、模糊、雜訊。"""
    x = im.astype(np.float32) / 255.0
    x = x ** random.uniform(0.55, 1.8)                                        # gamma
    x = (x - 0.5) * random.uniform(0.7, 1.35) + 0.5 + random.uniform(-0.15, 0.15)
    x = x * np.array([random.uniform(0.8, 1.2), random.uniform(0.9, 1.1), random.uniform(0.8, 1.2)], np.float32)  # 色溫
    h, w = x.shape[:2]
    if random.random() < 0.6:                                                 # 陰影漸層
        a = random.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        g = (np.cos(a) * (xx / w - 0.5) + np.sin(a) * (yy / h - 0.5)) * random.uniform(0.2, 0.7)
        x = x * (1 + g[..., None])
    if random.random() < 0.7:                                                 # 反光亮斑（膠帶反光）
        for _ in range(random.randint(3, 14)):
            cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
            ax, ay = random.randint(4, 28), random.randint(3, 14)
            m = np.zeros((h, w), np.float32)
            cv2.ellipse(m, (cx, cy), (ax, ay), random.uniform(0, 180), 0, 360, 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), random.uniform(1.5, 4))
            x = x + m[..., None] * random.uniform(0.4, 1.0)
    x = np.clip(x, 0, 1)
    if random.random() < 0.4:
        x = cv2.GaussianBlur(x, (0, 0), random.uniform(0.6, 1.8))
    if random.random() < 0.5:
        x = x + np.random.normal(0, random.uniform(0.005, 0.03), x.shape).astype(np.float32)
    out = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    if random.random() < 0.3:                                                 # 低解析度（深度模式的彩色是 640x360 放大）
        out = cv2.resize(cv2.resize(out, (640, 360)), (1280, 720))
    return out


idx = [i for i in range(len(items)) if i not in DROP_IMAGES]
random.shuffle(idx)
val = set(idx[:12])
n_img = n_box = 0
for i in idx:
    it = items[i]
    im = cv2.imread(os.path.join(ROOT, it["file"]))
    H, W = im.shape[:2]
    labels = []
    for j, (u, v, r, k) in enumerate(it["cands"]):
        k = RELABEL.get((i, j), k)
        if (i, j) in REMOVE or k == "?":
            continue
        if (i, j) in IGNORE:
            cv2.circle(im, (int(u), int(v)), int(r) + 8, (128, 128, 128), -1)
            continue
        labels.append("%d %.6f %.6f %.6f %.6f" % (CLS[k], u / W, v / H, 2 * r / W, 2 * r / H))
    split = "val" if i in val else "train"
    variants = [im] + ([photometric(im) for _ in range(7)] if split == "train" else [photometric(im) for _ in range(2)])
    for a, v_ in enumerate(variants):
        name = "%03d_%d" % (i, a)
        cv2.imwrite(os.path.join(D, "images", split, name + ".jpg"), v_, [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(os.path.join(D, "labels", split, name + ".txt"), "w").write("\n".join(labels))
        n_img += 1; n_box += len(labels)
open(os.path.join(D, "data.yaml"), "w", encoding="utf-8").write(
    "path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: O\n  1: X\n" % D.replace("\\", "/"))
print("影像 %d 張（驗證用原圖 %d 張 ×3）、標註框 %d 個" % (n_img, len(val), n_box))

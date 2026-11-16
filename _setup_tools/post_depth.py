"""量兩側支架的深度（離相機 mm），判斷是貼桌的腳座還是立柱。存 out/post_depth.jpg／.png。"""
import os, sys
import cv2, numpy as np
ROOT = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1"
sys.path.insert(0, os.path.join(ROOT, "Calib"))
from common import OrbbecCamera
cam = OrbbecCamera(depth=True)
try:
    img = cam.grab(skip=3, timeout=15)
    d = np.median(np.stack([cam.grab(depth=True, skip=1, timeout=15)[1] for _ in range(7)]), axis=0)
finally:
    cam.close()
out = os.path.join(ROOT, "Arena", "out")
cv2.imwrite(os.path.join(out, "post_depth.jpg"), img)
cv2.imwrite(os.path.join(out, "post_depth.png"), d.clip(0, 65535).astype("uint16"))
def med(x0, y0, x1, y1):
    w = d[y0:y1, x0:x1]; w = w[w > 0]
    return (round(float(np.median(w)), 1), round(float(np.percentile(w, 5)), 1), len(w)) if len(w) else None
print("board center", med(580, 335, 620, 375))
print("table left of board", med(440, 600, 470, 640), "table right", med(780, 600, 810, 640))
for y0 in (300, 330, 360, 390, 420, 450):
    print("L y=%d:" % y0, [(x, med(x, y0, x + 30, y0 + 30)) for x in range(0, 420, 30)])
for y0 in (330, 360, 390, 420, 450):
    print("R y=%d:" % y0, [(x, med(x, y0, x + 30, y0 + 30)) for x in range(960, 1280, 40)])

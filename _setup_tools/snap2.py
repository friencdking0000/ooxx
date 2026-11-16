import os, sys
R = os.path.join(os.environ["USERPROFILE"], "Desktop", "OOXX-robot-v1.1")
sys.path.insert(0, os.path.join(R, "Calib"))
import cv2, numpy as np
import align
from common import OrbbecCamera
c = OrbbecCamera(depth=True)
try:
    img = c.grab(skip=5, timeout=15)
    ds = [c.grab(depth=True, skip=1, timeout=15)[1] for _ in range(5)]
finally:
    c.close()
M = align.load()
d = align.apply(M, np.median(np.stack(ds), axis=0).astype(np.float32), depth=True)
out = os.path.join(R, "Arena", "out")
cv2.imwrite(os.path.join(out, "snap_now.jpg"), align.apply(M, img))
cv2.imwrite(os.path.join(out, "snap_now.png"), d.clip(0, 65535).astype(np.uint16))
print("ok", img.shape)

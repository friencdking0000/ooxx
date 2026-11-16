import sys, time, os
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1"
sys.path.insert(0, os.path.join(R, "Calib")); os.chdir(os.path.join(R, "Calib"))
import cv2, numpy as np
from common import OrbbecCamera
c = OrbbecCamera(); t = time.time()
for _ in range(10): img = c.grab(skip=0, timeout=10)
print("color 10 frames %.1fs" % (time.time() - t), img.shape); c.close(); time.sleep(0.5)
cv2.imwrite(os.path.join(R, "Arena", "cam_full.jpg"), img)
c = OrbbecCamera(depth=True); t = time.time()
ds = [c.grab(depth=True, skip=1, timeout=15)[1] for _ in range(5)]
c.close()
d = np.median(np.stack(ds), axis=0)
print("depth 5 frames %.1fs" % (time.time() - t), d.shape, "valid %.0f%%" % (100 * (d > 0).mean()), "board center depth %.0f mm" % np.median(d[340:380, 620:660][d[340:380, 620:660] > 0]))
np.save(os.path.join(R, "Arena", "depth_test.npy"), d)

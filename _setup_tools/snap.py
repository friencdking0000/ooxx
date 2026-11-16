import os, sys, glob, shutil
R = os.path.join(os.environ["USERPROFILE"], "Desktop", "OOXX-robot-v1.1")
sys.path.insert(0, os.path.join(R, "Calib"))
import cv2
from common import OrbbecCamera
c = OrbbecCamera(depth=False)
im = c.grab(skip=5, timeout=15)
c.close()
out = os.path.join(R, "Arena", "out")
cv2.imwrite(os.path.join(out, "after_grip.jpg"), im)
for i, f in enumerate(sorted(glob.glob(os.path.join(out, "grip_20261115_17*.jpg")))):
    shutil.copy(f, os.path.join(out, "g%d.jpg" % i)); print(i, os.path.basename(f))

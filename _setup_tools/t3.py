import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1"
sys.path.insert(0, os.path.join(R, "Calib")); os.chdir(os.path.join(R, "Calib"))
from pyorbbecsdk import Pipeline
p = Pipeline(); print("conn:", p.get_device().get_device_info().get_connection_type()); del p; time.sleep(0.5)
from common import OrbbecCamera
c = OrbbecCamera(depth=True, hires=True); t = time.time()
for i in range(15):
    img, d = c.grab(skip=0, timeout=8, depth=True)
print("15 frames %.1fs" % (time.time() - t), img.shape, d.shape, "valid %.0f%%" % (100 * (d > 0).mean()), "mean color %.0f" % img.mean())
c.close()

"""從對弈伺服器的相機程式（cam_worker）拿一組影像：彩色 3 張中位數＋深度約 5 張中位數（已對齊）。存成 <名稱>.jpg／<名稱>_depth.png。
    python grab_files.py <資料夾> <名稱>"""
import os, sys, time
import cv2, numpy as np
A = os.path.join(os.environ["USERPROFILE"], "Desktop", "OOXX-robot-v1.1", "Arena")
CAM, DEP, REQ = (os.path.join(A, n) for n in ("cam_full.jpg", "cam_depth.png", "depth_request"))
out, name = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
t0 = time.time()
open(REQ, "w").close()
ds, last = [], 0.0
while time.time() - t0 < 30:
    mt = os.path.getmtime(DEP) if os.path.exists(DEP) else 0.0
    if mt > t0 and mt != last:
        d = cv2.imread(DEP, cv2.IMREAD_UNCHANGED)
        if d is not None:
            ds.append(d.astype(np.float32)); last = mt
    if not os.path.exists(REQ) and (ds or time.time() - t0 > 3):
        break
    time.sleep(0.05)
ims, lastc = [], 0.0
while len(ims) < 3 and time.time() - t0 < 45:
    mt = os.path.getmtime(CAM)
    if mt > t0 and mt != lastc:
        im = cv2.imread(CAM)
        if im is not None:
            ims.append(im); lastc = mt
    time.sleep(0.05)
cv2.imwrite(os.path.join(out, name + ".jpg"), np.median(np.stack(ims), axis=0).astype(np.uint8))
if ds:
    cv2.imwrite(os.path.join(out, name + "_depth.png"), np.median(np.stack(ds), axis=0).astype(np.uint16))
print("color", len(ims), "depth", len(ds))

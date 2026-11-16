"""A 在 X 疊上試吸（不搬運）：每個偏移 碰到→多壓→開真空→原地抬 40 mm→拍 V4K 側面（OBS 視窗）→放回。"""
import json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Calib"))
import mover, stocks
from common import OrbbecCamera, enable, sync_move
from world import px_to_arm

def shot(name):
    p = os.path.join(HERE, "out", name + ".png")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(HERE, "obs_shot.ps1"), p], timeout=30)
    import cv2
    im = cv2.imread(p); h, w = im.shape[:2]
    cv2.imwrite(os.path.join(HERE, "out", name + ".jpg"), cv2.resize(im[int(h*0.2):int(h*0.42), int(w*0.24):int(w*0.5)], None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))

plan = json.load(open(os.path.join(os.path.dirname(HERE), "Calib", "cell_plan.json"), encoding="utf-8"))
cam = OrbbecCamera(depth=True)
img = cam.grab(skip=3, timeout=15); d = np.median(np.stack([cam.grab(depth=True, skip=1, timeout=15)[1] for _ in range(5)]), axis=0); cam.close()
M = mover.Mover()
try:
    A = M.arms["A"]
    st = [s for s in stocks.detect(img, d, M.arms["B"].cal, [plan[str(i)]["px"] for i in range(9)]) if s["kind"] == "X" and s["px"][1] > 450][0]
    c0 = px_to_arm(A.cal, st["px"], depth=st["top"]); z_top = st["z_touch"] + A.dz
    z_surf = mover.Z_TOUCH_B + (stocks.STD_TOP_DEPTH - st["table"]) + A.dz
    print("X 疊 %d 顆 影像 %s → A %s" % (st["n"], np.round(st["px"]), c0.round(1)), flush=True)
    enable(A.arm, speed=15); A.arm.dash.sendRecvMsg("SetCollisionLevel(3)")
    A.goto(c0); shot("grip_hover")
    R = M.T[0]
    for k, offB in enumerate([(-20, 0), (-12, -14), (-12, 14), (-27, 0), (-20, -15), (-20, 15)]):
        p = c0 + R.T @ np.array(offB, float)
        if not (A.ik(p, z_top - 12) and A.ik(p, -40)):
            print("偏移 %s：到不了" % (offB,), flush=True); continue
        sync_move(A.arm, "MovL(%f,%f,%f,%f)" % (p[0], p[1], -40.0, A.r))
        zc = A.touch(p, z_top, vacuum_on=True, below=11.0, z_surf=z_surf)
        if zc is None:
            print("偏移 %s：沒碰到／太低" % (offB,), flush=True); A.lift(); continue
        time.sleep(0.5)
        q = A.arm.pose()
        sync_move(A.arm, "MovL(%f,%f,%f,%f)" % (q[0], q[1], q[2] + 40.0, A.r)); time.sleep(1.0)
        shot("grip_%d" % k)
        sync_move(A.arm, "MovL(%f,%f,%f,%f)" % (q[0], q[1], zc + 2.0, A.r))
        A.arm.dash.sendRecvMsg("DOExecute(1,0)"); time.sleep(0.8)
        A.lift()
        print("偏移 %s：接觸 Z=%.1f，已拍 grip_%d" % (offB, zc, k), flush=True)
    A.go_park()
except BaseException as e:
    print("停止：", repr(e), flush=True); M.park_all(); raise
finally:
    M.close()

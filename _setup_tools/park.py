import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1"
for d in ("Arena", "Calib", "MG400"): sys.path.insert(0, os.path.join(R, d))
os.chdir(os.path.join(R, "Arena"))
import numpy as np
import mover
from common import enable
go = "--go" in sys.argv
M = mover.Mover()
try:
    for name in ("A", "B"):
        a = M.arms[name]
        q = a.arm.pose()
        rin = a.route_rin(np.array(q[:2]), a.park)
        print(name, "mode", a.arm.mode(), "pose", [round(x, 1) for x in q[:4]], "park", a.park.round(1), "route rin", rin, flush=True)
        if not go:
            continue
        enable(a.arm, speed=8)
        a.arm.dash.sendRecvMsg("SetCollisionLevel(3)")
        if rin is None:                      # 縮得太靠近底座：先沿目前方向直線往外伸到半徑 220 再走正常路線
            from common import sync_move
            p0 = np.array(q[:2]); out = p0 / np.hypot(*p0) * 220.0
            if not (a.ik(out, q[2]) and a.ik(out, -40.0)) or a.post(out, -40.0) or a.route_rin(out, a.park) is None:
                print("  往外伸也沒有安全路線，不動"); a.arm.dash.DisableRobot(); continue
            sync_move(a.arm, "MovL(%f,%f,%f,%f)" % (out[0], out[1], q[2], a.r))
            print("  先伸到", out.round(1), flush=True)
        a.goto(a.park)
        print("  parked", [round(x, 1) for x in a.arm.pose()[:3]], "mode", a.arm.mode(), flush=True)
        a.arm.dash.sendRecvMsg("DOExecute(1,0)")
        a.arm.dash.DisableRobot()
finally:
    M.close()

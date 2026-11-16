import json, os, sys
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1"
for d in ("Arena", "Calib", "MG400"): sys.path.insert(0, os.path.join(R, d))
os.chdir(os.path.join(R, "Arena"))
import numpy as np, cv2
import mover, stocks
from common import OrbbecCamera
from play import Z_HOVER
plan = json.load(open(r"..\Calib\cell_plan.json", encoding="utf-8"))
cam = OrbbecCamera(depth=True)
try:
    img = cam.grab(skip=3, timeout=15)
    d = np.median(np.stack([cam.grab(depth=True, skip=1, timeout=15)[1] for _ in range(5)]), axis=0)
finally:
    cam.close()
cv2.imwrite("out/diag_img.png", img)
M = mover.Mover()
try:
    b = M.arms["B"]
    S = stocks.detect(img, d, b.cal, [plan[str(i)]["px"] for i in range(9)])
    for s in S:
        print(s["kind"], np.round(s["px"]), "n", s["n"], "xy", np.round(s["xy"], 1), "z_touch %.1f" % s["z_touch"], "top", s["top"], "table", s["table"])
    for st in [s for s in S if s["kind"] == "O"]:
        ok = []
        for cell in range(9):
            c = np.array(plan[str(cell)]["B"]); zd = mover.Z_TOUCH_B + 1.5 + mover.board_dz(cell)
            pl = M.plan(st["xy"], st["z_touch"], c, zd, "O", None)
            ok.append(cell if pl else None)
        print("O", np.round(st["px"]), "可下格:", ok)
        sl = np.asarray(st["xy"]); zs = st["z_touch"]
        print("  在料疊正上方: post", b.post(sl, zs), b.post(sl, Z_HOVER), "leg park->src", b.leg_ok(b.park, sl), "ik zs-9", b.ik(sl, zs - 9), "ik zs", b.ik(sl, zs), "ik hover", b.ik(sl, Z_HOVER), "r=%.0f" % np.hypot(*sl))
        print("  route_rin", b.route_rin(b.park, sl), "park", b.park)
finally:
    M.close()

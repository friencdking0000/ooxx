"""只算不動：棋盤右側外的轉送點候選（B 座標），兩台都要搆得到（IK、支架、路線），印支架餘量、桌面高度、離格 5 多遠。"""
import os, sys, json
R = os.path.join(os.environ["USERPROFILE"], "Desktop", "OOXX-robot-v1.1")
sys.path.insert(0, os.path.join(R, "Arena")); sys.path.insert(0, os.path.join(R, "Calib"))
os.chdir(os.path.join(R, "Arena"))
import numpy as np
import grip_test as G
import mover
from common import OrbbecCamera

fit = json.load(open(G.FIT, encoding="utf-8")); plan = json.load(open(G.PLAN, encoding="utf-8"))
cam = OrbbecCamera(depth=True); M = mover.Mover()
try:
    sc = G.Scene(cam, plan, fit)
    sc.calB = M.arms["B"].cal
    _, piles = sc.measure(sc.calB)
    sc.use_cam_posts()
    c5 = np.array(fit["B"]["cells"]["5"][:2], float)
    c4 = np.array(fit["B"]["cells"]["4"][:2], float)
    u = (c5 - c4) / np.hypot(*(c5 - c4))                    # 往畫面右（格 4 → 格 5）
    v = np.array([u[1], -u[0]])
    for along in (80.0, 90.0, 100.0, 110.0):
        for side in (0.0, -30.0, 30.0):
            hB = c5 + along * u + side * v
            row = []
            for n in ("A", "B"):
                arm = M.arms[n]
                h = sc.arm_from_cam(n, sc.cam_xy("B", hB))
                zs = sc.surface_z(n, h)
                why = G.blocked(arm, [("轉送點", h, zs + G.PIECE - 6)]) if zs is not None else ["量不到高度"]
                post, mg = sc.post_margin(n, h)
                row.append("%s %s 半徑 %.0f 桌面 Z %.1f 離%s %.0f mm %s" % (n, h.round(1), np.hypot(*h), zs if zs else 0, post, mg, "可以" if not why else "、".join(why)))
            px = sc.px_of("B", hB)[:2]
            occ = [s for s in piles if np.hypot(*(s["arm_xy"]["B"] - hB)) < 60]
            print("B %s（格 5 右 %.0f、偏 %.0f；影像 %s）%s\n   %s\n   %s" % (hB.round(1), along, side, px.round(0),
                  "有東西" if occ else "空", row[0], row[1]), flush=True)
finally:
    M.close(); cam.close()

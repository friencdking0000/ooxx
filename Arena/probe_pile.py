"""觸覺找料疊中心（相機看不到手臂在料區上方的位置時用）：手臂在料疊周圍慢慢往下點，接觸高度掉到桌面＝超出邊緣。
每個方向二分找邊緣 → 料疊在「這台手臂座標」裡的真正中心，跟影像＋校正算的差多少。

    python probe_pile.py A X        # A 臂探 X 疊；結果存 pile_fix_A.json（dx, dy：真正中心 − 校正算的中心）
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Calib"))
import mover                                   # noqa: E402
import stocks                                  # noqa: E402
from common import OrbbecCamera, enable       # noqa: E402
from play import Z_HOVER                       # noqa: E402
from world import px_to_arm                    # noqa: E402

R_PIECE = 42.0


def main():
    name, kind = sys.argv[1], sys.argv[2]
    plan = json.load(open(os.path.join(os.path.dirname(HERE), "Calib", "cell_plan.json"), encoding="utf-8"))
    cam = OrbbecCamera(depth=True)
    try:
        img = cam.grab(skip=3, timeout=15)
        d = np.median(np.stack([cam.grab(depth=True, skip=1, timeout=15)[1] for _ in range(5)]), axis=0)
    finally:
        cam.close()
    M = mover.Mover()
    try:
        arm = M.arms[name]
        S = [s for s in stocks.detect(img, d, M.arms["B"].cal, [plan[str(i)]["px"] for i in range(9)]) if s["kind"] == kind]
        S = [s for s in S if arm.leg_ok(arm.park, arm.loc(s["xy"]))]
        if not S:
            raise SystemExit("沒有 %s 搆得到的 %s 疊" % (name, kind))
        st = max(S, key=lambda s: s["n"])
        c0 = px_to_arm(arm.cal, st["px"], depth=st["top"]) if name == "A" else np.asarray(st["xy"], float)
        z_top = st["z_touch"] + arm.dz
        z_tab = mover.Z_TOUCH_B + (stocks.STD_TOP_DEPTH - st["table"]) + arm.dz
        print("%s 疊 %d 顆，影像 %s → %s 座標 %s；頂面約 Z=%.1f、桌面約 %.1f" % (kind, st["n"], np.round(st["px"]), name, c0.round(1), z_top, z_tab), flush=True)
        enable(arm.arm, speed=15)
        arm.arm.dash.sendRecvMsg("SetCollisionLevel(3)")
        arm.goto(c0)

        def probe(p):
            """回傳 True＝這點在料疊上、False＝桌面、None＝到不了。"""
            if not (arm.ik(p, z_tab - 3) and arm.ik(p, Z_HOVER)) or arm.post(p, Z_HOVER):
                print("   點 %s 到不了（IK／支架），半徑 %.0f" % (np.round(p - c0, 0), np.hypot(*p)), flush=True)
                return None
            from common import sync_move
            sync_move(arm.arm, "MovL(%f,%f,%f,%f)" % (p[0], p[1], z_top + 12, arm.r))
            zc = arm.touch(p, z_top, vacuum_on=False, over=8.0, below=(z_top - z_tab) + 3.0)
            sync_move(arm.arm, "MovL(%f,%f,%f,%f)" % (p[0], p[1], z_top + 12, arm.r))
            on = zc is not None and (zc + arm.COMPRESS - z_tab) > (z_top - z_tab) * 0.5
            print("   點 %s → 接觸 %s → %s" % (np.round(p - c0, 0), None if zc is None else round(zc, 1), "疊上" if on else "桌面"), flush=True)
            return on

        if not probe(c0):
            raise SystemExit("校正算的中心點下去不是料疊，偏差超過一個棋子半徑")
        edges = {}
        for key, u in (("+x", (1, 0)), ("-x", (-1, 0)), ("+y", (0, 1)), ("-y", (0, -1))):
            u = np.array(u, float)
            lo, hi = 0.0, 60.0                       # lo 在疊上、hi 在桌面
            r_hi = probe(c0 + hi * u)
            if r_hi is None:
                hi, r_hi = 48.0, probe(c0 + 48.0 * u)
            if r_hi is None or r_hi:                 # 到不了，或 60 mm 外還是疊（碰到隔壁那疊）：這個方向不用
                continue
            for _ in range(3):
                mid = (lo + hi) / 2
                r = probe(c0 + mid * u)
                if r is None:
                    break
                lo, hi = (mid, hi) if r else (lo, mid)
            edges[key] = (lo + hi) / 2
            print(" 邊緣 %s：%.1f mm" % (key, edges[key]), flush=True)
        fix = np.zeros(2)
        for i, (p_, n_) in enumerate((("+x", "-x"), ("+y", "-y"))):
            if p_ in edges and n_ in edges:
                fix[i] = (edges[p_] - edges[n_]) / 2
            elif p_ in edges:
                fix[i] = edges[p_] - R_PIECE
            elif n_ in edges:
                fix[i] = R_PIECE - edges[n_]
        print("結果：料疊真正中心 = 校正算的中心 + %s mm（%s 座標）" % (fix.round(1), name), flush=True)
        json.dump({"dx": float(fix[0]), "dy": float(fix[1]), "edges": edges, "t": time.time()},
                  open(os.path.join(HERE, "pile_fix_%s.json" % name), "w"))
        arm.lift()
        arm.go_park()
    except BaseException as e:
        print("停止：", repr(e), flush=True)
        M.park_all()
        raise
    finally:
        M.close()


if __name__ == "__main__":
    main()

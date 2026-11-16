"""A 轉送：B 搆不到的棋子，由 A 吸起、挪到「B 吸得到、A 也放得到」的空位，之後 B 再接手。

    python a_relay.py <B座標中心x,y> <頂面比格4標準高 dz> [--dry]
例：python a_relay.py 410.9,-76.2 -0.2 --dry

高度：A 碰到「格 4 標準棋子頂面」約 Z=-151.82（兩臂方盒高度差 3 mm 推得，test_A_pick 實測可吸）。
A 吸取點可以偏離中心（棋子貼了膠帶），偏移用 A 自己可達為準；放下時棋子平移，吸盤相對棋子的位置不變。
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import connect_arm, enable, sync_move                       # noqa: E402
from pickplace import STOCK, PIECE, Z_TOUCH_BOARD, counts, seg_clear, travel_points   # noqa: E402
from plan_cells import hits_post                                         # noqa: E402
from play import PARK, Z_HOVER, a_to_b_transform, approach, go_park, path_clear       # noqa: E402
from world import load                                                   # noqa: E402

Z_TOUCH_A = Z_TOUCH_BOARD + 3.0          # ≈ -151.82
PRESS_A = 1.0
RELEASE_UP = 3.0                         # 放下時棋子底離棋盤約 3 mm 放開（棋盤高低不平，留餘裕）
B_PRESS = 1.0


def main():
    dry = "--dry" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cB = np.array([float(v) for v in args[0].split(",")])
    dz = float(args[1])
    R, t = a_to_b_transform()
    toA = lambda p: R.T @ (np.asarray(p, float) - t)       # noqa: E731
    calA, calB = load("A"), load("B")
    dstP = np.array(STOCK["stocks"]["P疊"]["xy"])
    z_place_P = STOCK["stocks"]["P疊"]["z_first"] + counts()["P疊"] * PIECE
    parkB, parkA = np.array(PARK["B"][:2]), np.array(PARK["A"][:2])
    zA_pick = Z_TOUCH_A + dz - PRESS_A
    zA_rel = Z_TOUCH_A + dz + RELEASE_UP
    zB_pick = Z_TOUCH_BOARD + dz - B_PRESS
    A, B = connect_arm("A"), connect_arm("B")
    try:
        A.wait_feed(), B.wait_feed()
        rA, rB = float(A.pose()[3]), float(B.pose()[3])
        ikA = lambda p, z: A.dash.InverseSolution(p[0], p[1], z, rA, 0, 0).startswith("0,")   # noqa: E731
        ikB = lambda p, z: B.dash.InverseSolution(p[0], p[1], z, rB, 0, 0).startswith("0,")   # noqa: E731

        # 1) B 可接手的新位置 Q（棋子中心，B 座標）：B 吸洞中心、路線、支架都要過
        def b_ok(q):
            if hits_post(calB, q, zB_pick) or hits_post(calB, q, Z_HOVER):
                return False
            if not (seg_clear(calB, travel_points(parkB, q), Z_HOVER) and seg_clear(calB, travel_points(q, dstP), Z_HOVER)
                    and seg_clear(calB, travel_points(dstP, parkB), Z_HOVER)):
                return False
            return ikB(q, zB_pick) and ikB(q, Z_HOVER) and ikB(dstP, z_place_P) and ikB(dstP, Z_HOVER)

        Qs = sorted([cB + (dx, dy) for dx in np.arange(-60, 1, 5) for dy in np.arange(0, 91, 5)],
                    key=lambda q: np.hypot(*(q - cB)))
        # 2) A 的吸取偏移 off（B 座標，相對棋子中心）：A 吸取、放下（Q+off）都可達、路線不碰支架
        offs = sorted([np.array((dx, dy), float) for dx in np.arange(-24, 25, 3) for dy in np.arange(-24, 25, 3)
                       if np.hypot(dx, dy) <= 24], key=lambda o: np.hypot(*o))
        plan = None
        for off in offs:
            pA = toA(cB + off)
            if not (ikA(pA, zA_pick) and ikA(pA, Z_HOVER) and path_clear("A", PARK["A"], pA)) \
                    or hits_post(calA, pA, zA_pick) or hits_post(calA, pA, Z_HOVER):
                continue
            for q in Qs:
                if np.hypot(*(q - cB)) < 25:                     # 至少挪 25 mm，不然等於沒動
                    continue
                qA = toA(q + off)
                seg = [pA + s * (qA - pA) for s in np.linspace(0, 1, 15)]
                if any(hits_post(calA, w, Z_HOVER) for w in seg) or hits_post(calA, qA, zA_rel):
                    continue
                if not (ikA(qA, zA_rel) and ikA(qA, Z_HOVER)):
                    continue
                if b_ok(q):
                    plan = (off, pA, q, qA)
                    break
            if plan:
                break
        if plan is None:
            raise SystemExit("找不到 A 吸得到、B 又接得到的轉送方案")
        off, pA, q, qA = plan
        print("A 吸取：棋子中心偏 %s mm（A 座標 %s，Z=%.1f）→ 挪到棋子中心 B %s（移 %.0f mm），A 放開 Z=%.1f"
              % (off.round(1), pA.round(1), zA_pick, q.round(1), np.hypot(*(q - cB)), zA_rel))
        print("之後 B 在 %s 吸洞中心（Z=%.1f）→ 備用疊" % (q.round(1), zB_pick))
        json.dump({"q": q.tolist(), "dz": dz}, open(os.path.join(HERE, "out", "relay_target.json"), "w"))
        if dry:
            return
        enable(A, speed=10)
        p = A.pose()
        if p[2] < Z_HOVER - 1:
            sync_move(A, "MovL(%f,%f,%f,%f)" % (p[0], p[1], Z_HOVER, rA))
        approach(A, pA)
        sync_move(A, "MovL(%f,%f,%f,%f)" % (pA[0], pA[1], zA_pick, rA))
        print("  A 吸：", A.dash.sendRecvMsg("DOExecute(1,1)").strip())
        time.sleep(0.8)
        sync_move(A, "MovL(%f,%f,%f,%f)" % (pA[0], pA[1], Z_HOVER, rA))
        sync_move(A, "MovL(%f,%f,%f,%f)" % (qA[0], qA[1], Z_HOVER, rA))
        sync_move(A, "MovL(%f,%f,%f,%f)" % (qA[0], qA[1], zA_rel, rA))
        print("  A 放：", A.dash.sendRecvMsg("DOExecute(1,0)").strip())
        time.sleep(0.6)
        sync_move(A, "MovL(%f,%f,%f,%f)" % (qA[0], qA[1], Z_HOVER, rA))
        go_park(A)
        print("A 回停放", A.pose().round(1))
    except BaseException as e:
        print("停止：", e)
        try:
            A.dash.ResetRobot()
        except Exception:
            pass
        raise
    finally:
        for arm in (A, B):
            try:
                arm.dash.DisableRobot()
            except Exception:
                pass
            arm.close()


if __name__ == "__main__":
    main()

"""測試 A 的吸取：B 從 O 疊拿一顆放到格 0 附近 → A 吸起（推算高度）→ 相機確認 → A 放回原位。

A 的吸取高度由 B 推算（兩台吸盤相同，使用者確認）：B 在 Z=-40 時方盒頂面高於棋盤 273 mm、A 270 mm，
B 碰到棋子頂面 Z=-154.82 → A 約 Z=-151.82。
測試位置用格 0：A 從停放點到格 4（中間）的路線會經過左側支架（2026-11-14 擦過），格 0 的路線已檢查安全。
    python test_A_pick.py --dry     # 只檢查
    python test_A_pick.py           # 執行（速度 10%）
"""
import json
import os
import sys
import time

import numpy as np

from common import Camera, connect_arm, enable, sync_move
from pickplace import STOCK, Z_TOUCH_BOARD, counts, seg_clear, travel, travel_points
from plan_cells import hits_post
from play import PARK, Z_HOVER, a_to_b_transform
from world import load, px_to_arm

HERE = os.path.dirname(os.path.abspath(__file__))
DRY = "--dry" in sys.argv
Z_TOUCH_A = Z_TOUCH_BOARD + (273.0 - 270.0)          # ≈ -151.82
PRESS = 1.0


def find_piece(near_px, r=140):
    """俯視畫面中 near_px 附近的棋子（深色圓盤）中心像素。"""
    import cv2
    cam = Camera()
    try:
        img = cam.grab(skip=3, timeout=20)
    finally:
        cam.close()
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    x0, y0 = max(0, int(near_px[0]) - r), max(0, int(near_px[1]) - r)
    sub = g[y0:int(near_px[1]) + r, x0:int(near_px[0]) + r]
    cs = cv2.HoughCircles(sub, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60, param1=80, param2=30, minRadius=40, maxRadius=70)
    if cs is None:
        return None
    best = min(cs[0], key=lambda c: np.hypot(c[0] + x0 - near_px[0], c[1] + y0 - near_px[1]))
    cx, cy = best[0] + x0, best[1] + y0
    if g[int(cy) - 15:int(cy) + 15, int(cx) - 15:int(cx) + 15].mean() > 110:
        return None                                      # 不夠暗，不是棋子
    return np.array([cx, cy])


def check(arm, cal, name, pts_z, legs):
    r = float(arm.pose()[3])
    for xy, z in pts_z:
        if not arm.dash.InverseSolution(xy[0], xy[1], z, r, 0, 0).startswith("0,"):
            raise SystemExit("%s IK 不可達：%s Z=%.1f（半徑 %.0f）" % (name, np.round(xy, 1), z, np.hypot(*xy)))
        if hits_post(cal, xy, z):
            raise SystemExit("%s 會碰支架：%s Z=%.1f" % (name, np.round(xy, 1), z))
    for a, b in legs:
        if not seg_clear(cal, travel_points(a, b), Z_HOVER):
            raise SystemExit("%s 路線會碰支架：%s → %s" % (name, np.round(a, 1), np.round(b, 1)))


def main():
    plan = json.load(open(os.path.join(HERE, "cell_plan.json"), encoding="utf-8"))
    calA, calB = load("A"), load("B")
    # ---- 1. B：O 疊頂端 → 格 0 附近（往外偏到 B 放置高度可達）----
    n = counts()["O疊"]
    if n < 1:
        raise SystemExit("O 疊是空的")
    stk = np.array(STOCK["stocks"]["O疊"]["xy"])
    z_stack_top = STOCK["stocks"]["O疊"]["z_first"] - 1.5 + (n - 1) * STOCK["piece_mm"]   # 頂端那顆的頂面
    z_pick_B = z_stack_top - 5.0                                                # 疊上吸取多壓 5 mm（實測 2 mm 吸不到，使用者：再壓 2～3 mm）
    z_place_board = Z_TOUCH_BOARD + 1.5                                         # 棋子底離板 1.5 mm 放開
    c0 = np.array(plan["0"]["B"])
    u = c0 / np.hypot(*c0)
    B = connect_arm("B")
    A = connect_arm("A")
    try:
        B.wait_feed(), A.wait_feed()
        rB = float(B.pose()[3])
        tgt = None
        for d in np.arange(0, 20.5, 1.0):                                       # 從格 0 中心往外找 B 可放的點
            cand = c0 + d * u
            if all(B.dash.InverseSolution(cand[0], cand[1], z, rB, 0, 0).startswith("0,") for z in (z_place_board, Z_HOVER)):
                tgt = cand
                break
        targ = next((a[6:] for a in sys.argv if a.startswith("--tgt=")), None)
        if targ:                                         # 指定放置點（B 座標）：要同時讓 A 的路線避開支架
            tgt = np.array([float(v) for v in targ.split(",")])
        if tgt is None:
            raise SystemExit("格 0 附近找不到 B 可放的點")
        print("B：O 疊（第 %d 顆，吸取 Z=%.1f）→ 格 0 附近 %s（離格中心 %.0f mm，放開 Z=%.1f）"
              % (n, z_pick_B, tgt.round(1), np.hypot(*(tgt - c0)), z_place_board))
        parkB = np.array(PARK["B"][:2])
        check(B, calB, "B", [(stk, z_pick_B), (stk, Z_HOVER), (tgt, z_place_board), (tgt, Z_HOVER)],
              [(B.pose()[:2], stk), (stk, tgt), (tgt, parkB)])
        # A 的檢查要等棋子實際位置；先用目標點預檢
        T = a_to_b_transform()
        R, t = T
        tgtA = R.T @ (tgt - t)
        z_pick_A = Z_TOUCH_A - PRESS
        parkA = np.array(PARK["A"][:2])
        from play import path_clear
        check(A, calA, "A", [(tgtA, z_pick_A), (tgtA, Z_HOVER)], [])
        if not path_clear("A", PARK["A"], tgtA):        # A 用 play.approach／go_park 的路線（停放點直接 MovJ 到目標方向小半徑點）
            raise SystemExit("A 路線會碰支架")
        print("A：預計在 %s 吸取（Z=%.1f），檢查通過" % (tgtA.round(1), z_pick_A))
        if DRY:
            return

        # ---- 執行 B ----
        enable(B, speed=10)
        p = B.pose()
        if p[2] < Z_HOVER - 1:
            sync_move(B, "MovL(%f,%f,%f,%f)" % (p[0], p[1], Z_HOVER, rB))
        travel(B, stk, rB)
        sync_move(B, "MovL(%f,%f,%f,%f)" % (stk[0], stk[1], z_pick_B, rB))
        B.dash.sendRecvMsg("DOExecute(1,1)")
        time.sleep(0.8)
        sync_move(B, "MovL(%f,%f,%f,%f)" % (stk[0], stk[1], Z_HOVER, rB))
        travel(B, tgt, rB)
        sync_move(B, "MovL(%f,%f,%f,%f)" % (tgt[0], tgt[1], z_place_board, rB))
        B.dash.sendRecvMsg("DOExecute(1,0)")
        time.sleep(0.6)
        sync_move(B, "MovL(%f,%f,%f,%f)" % (tgt[0], tgt[1], Z_HOVER, rB))
        travel(B, parkB, rB)
        B.dash.DisableRobot()
        c = counts()
        c["O疊"] -= 1
        json.dump(c, open(os.path.join(HERE, "stock_count.json"), "w", encoding="utf-8"), ensure_ascii=False)
        print("B 已放好並回停放；O 疊剩 %d 顆" % c["O疊"])

        # ---- 找棋子實際中心 → A 座標 ----
        from world import arm_to_px
        near = arm_to_px(calB, tgt, depth=535)
        pc = find_piece(near)
        if pc is None:
            raise SystemExit("找不到 B 放的棋子（可能沒吸住或掉了），停止")
        pieceA = px_to_arm(calA, pc, depth=535)
        print("棋子實際中心：影像 %s → A 座標 %s（與預期差 %.1f mm）" % (pc.round(0), pieceA.round(1), np.hypot(*(pieceA - tgtA))))
        check(A, calA, "A", [(pieceA, z_pick_A), (pieceA, Z_HOVER)], [])
        if not path_clear("A", PARK["A"], pieceA):
            raise SystemExit("A 路線會碰支架")

        # ---- 執行 A：吸起 → 確認 → 放回 ----
        rA = float(A.pose()[3])
        enable(A, speed=10)
        p = A.pose()
        if p[2] < Z_HOVER - 1:
            sync_move(A, "MovL(%f,%f,%f,%f)" % (p[0], p[1], Z_HOVER, rA))
        from play import approach, go_park
        approach(A, pieceA)                               # 停放點 → MovJ 到目標方向小半徑點 → MovL 伸出
        sync_move(A, "MovL(%f,%f,%f,%f)" % (pieceA[0], pieceA[1], Z_TOUCH_A + 3, rA))   # 先停在預估接觸上方 3 mm
        sync_move(A, "MovL(%f,%f,%f,%f)" % (pieceA[0], pieceA[1], z_pick_A, rA))
        A.dash.sendRecvMsg("DOExecute(1,1)")
        time.sleep(0.8)
        sync_move(A, "MovL(%f,%f,%f,%f)" % (pieceA[0], pieceA[1], Z_HOVER, rA))
        a0 = np.arctan2(pieceA[1], pieceA[0])
        from play import R_IN
        sync_move(A, "MovL(%f,%f,%f,%f)" % (R_IN * np.cos(a0), R_IN * np.sin(a0), Z_HOVER, rA))   # 讓開給相機看
        still = find_piece(near)
        got = still is None
        print("A 吸取結果：%s" % ("✓ 吸起來了（原位已空）" if got else "✗ 沒吸住（棋子還在原位）"))
        if got:
            sync_move(A, "MovL(%f,%f,%f,%f)" % (pieceA[0], pieceA[1], Z_HOVER, rA))   # 沿同一方向直線伸回
            sync_move(A, "MovL(%f,%f,%f,%f)" % (pieceA[0], pieceA[1], Z_TOUCH_A + 1.5, rA))
            A.dash.sendRecvMsg("DOExecute(1,0)")
            time.sleep(0.6)
            sync_move(A, "MovL(%f,%f,%f,%f)" % (pieceA[0], pieceA[1], Z_HOVER, rA))
            print("A 已把棋子放回原位")
        else:
            A.dash.sendRecvMsg("DOExecute(1,0)")
        go_park(A)                                         # 直線縮回小半徑 → MovJ 回停放點
        print("A 回停放 %s" % A.pose().round(1))
    except BaseException as e:
        print("停止：", e)
        for x in (A, B):
            try:
                x.dash.sendRecvMsg("DOExecute(1,0)")
                x.dash.ResetRobot()
            except Exception:
                pass
        raise
    finally:
        for x in (A, B):
            try:
                x.dash.DisableRobot()
            except Exception:
                pass
            x.close()


if __name__ == "__main__":
    main()

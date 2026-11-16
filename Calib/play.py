"""井字棋實際下一步：相機讀盤面 → minimax 選步（只選有手臂夠得到的格）→ 碰撞檢查 → 指派手臂落子（DO 代替吸盤）。

    python play.py                    # 乾跑：印盤面、選步、指派、碰撞檢查，不動
    python play.py --go               # 實際執行（一次只動一台）
    python play.py --go --cell=4      # 指定格子（測試用）
    python play.py --go --do=1        # 吸盤測試用的 DO 編號（預設 1）
高度：懸停 Z_HOVER=-40（校正高度）、放置 Z_PLACE=-90（實測：B 曾停在 Z=-114.5 於中央棋子上方未碰觸）。
吸盤真空尚未接，DO 開 0.5 s 再關，只是驗證訊號與流程；不會真的拿起棋子。
"""
import json
import os
import sys
import time

import numpy as np

from board import find_grid_robust
from common import Camera, connect_arm, enable, sync_move
from game import best_move, describe, whose_turn, winner
from world import load

HERE = os.path.dirname(os.path.abspath(__file__))
Z_HOVER, Z_PLACE = -40.0, -90.0
# 只用畫面左側的停放點：右側停放點 A(60,-340) 在 2026-11-10 疑似掃到相機支架、把相機撞偏（右側有相機支架橫桿）
PARKS = {"A": [(60.0, 340.0, Z_HOVER)],
         "B": [(0.0, -220.0, Z_HOVER)]}
PARK = {k: v[0] for k, v in PARKS.items()}                              # 目前選用（main 會依目標更新）
CLEAR_MM = 150.0              # 兩臂（TCP 與連桿）中心線最小間距：方盒約 45 mm 寬＋連桿寬＋餘裕
R_IN = 215.0                  # mm：先在這個小半徑轉向目標方向，再直線伸出（避開左右支架；2026-11-14）
EXCLUDE_CELLS = set()         # 不寫死：由支架禁入區、IK、兩臂淨空自動決定


def in_keepout(arm, xy):
    """目標在懸停或放置高度時，方盒（含半寬餘裕）是否進入左右支架範圍（plan_cells.POSTS）。"""
    from plan_cells import hits_post
    cal = load(arm)
    return bool(hits_post(cal, xy, Z_HOVER) or hits_post(cal, xy, Z_PLACE))


def radial_path(p0, p1, r_in=R_IN, n=30):
    """實際走法：p0 →（MovJ，半徑與角度同時插值）→ (r_in, 目標角) →（MovL）→ p1。回傳取樣點。"""
    a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
    r0, r1 = np.hypot(*p0[:2]), np.hypot(*p1[:2])
    pts = [((1 - t) * r0 + t * r_in) * np.array([np.cos((1 - t) * a0 + t * a1), np.sin((1 - t) * a0 + t * a1)])
           for t in np.linspace(0, 1, n)]
    pts += [((1 - t) * r_in + t * r1) * np.array([np.cos(a1), np.sin(a1)]) for t in np.linspace(0, 1, n)]
    return pts


def path_clear(arm, p0, p1):
    from plan_cells import hits_post
    cal = load(arm)
    return not any(hits_post(cal, q, Z_HOVER) for q in radial_path(p0, p1))


def guard_camera():
    """相機被移動過（CAMERA_MOVED 存在）時拒絕執行：手眼校正已失效。"""
    f = os.path.join(HERE, "CAMERA_MOVED")
    if os.path.exists(f):
        raise SystemExit("拒絕執行：" + open(f, encoding="utf-8").read().strip())


def opt(name, default):
    for a in sys.argv[1:]:
        if a.startswith("--%s=" % name):
            return type(default)(a.split("=", 1)[1])
    return default


def to_B(arm, xy, T):
    """手臂 arm 座標的 XY → B 座標。T = (R, t)：A→B 的剛體轉換。"""
    if arm == "B":
        return np.asarray(xy, float)
    R, t = T
    return R @ np.asarray(xy, float) + t


def seg_dist(p0, p1, q0, q1):
    best = 1e9
    for p, (a, b) in ((p0, (q0, q1)), (p1, (q0, q1)), (q0, (p0, p1)), (q1, (p0, p1))):
        d = b - a
        s = np.clip(((p - a) @ d) / (d @ d + 1e-9), 0, 1)
        best = min(best, float(np.linalg.norm(p - (a + s * d))))
    return best


def a_to_b_transform():
    """用兩臂校正把影像上同一組點換到兩個座標，求 A→B 剛體轉換。"""
    from world import px_to_arm, rigid2d
    A, B = load("A"), load("B")
    grid = [(x, y) for x in range(300, 1000, 100) for y in range(100, 700, 100)]
    pa = [px_to_arm(A, p) for p in grid]
    pb = [px_to_arm(B, p) for p in grid]
    R, t, _ = rigid2d(pa, pb)
    return R, t


def clearance(active, target_xy, T, park=None):
    """active 臂從底座到目標（含懸停）vs 另一臂停在 park 的線段，最小距離（B 座標）。"""
    other = "B" if active == "A" else "A"
    park = park or PARK[other]
    base = {"A": to_B("A", (0, 0), T), "B": np.zeros(2)}
    a0, a1 = base[active], to_B(active, target_xy, T)
    o0, o1 = base[other], to_B(other, park[:2], T)
    return seg_dist(a0, a1, o0, o1)


def best_park(active, target_xy, T):
    other = "B" if active == "A" else "A"
    return max(PARKS[other], key=lambda p: clearance(active, target_xy, T, p))


def at(arm, xyz, tol=3.0):
    return np.abs(arm.pose()[:3] - np.asarray(xyz)).max() < tol


def go_park(arm):
    """先只升不降到 Z_HOVER，若在外側就沿目前方向直線縮回 R_IN，再 MovJ 回停放點（approach 的反向）。"""
    p = arm.pose()
    r = float(p[3])
    if p[2] < Z_HOVER - 1:
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (p[0], p[1], Z_HOVER, r))
    rad = float(np.hypot(p[0], p[1]))
    if rad > R_IN + 5 and not at(arm, PARK[arm.name]):
        a = np.arctan2(p[1], p[0])
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (R_IN * np.cos(a), R_IN * np.sin(a), Z_HOVER, r))
    x, y, z = PARK[arm.name]
    sync_move(arm, "MovJ(%f,%f,%f,%f)" % (x, y, z, r))


def approach(arm, txy):
    """從停放點出發：MovJ 到 (R_IN, 目標方向) → MovL 直線伸到目標上方（Z_HOVER）。"""
    r = float(arm.pose()[3])
    a = np.arctan2(txy[1], txy[0])
    sync_move(arm, "MovJ(%f,%f,%f,%f)" % (R_IN * np.cos(a), R_IN * np.sin(a), Z_HOVER, r))
    sync_move(arm, "MovL(%f,%f,%f,%f)" % (txy[0], txy[1], Z_HOVER, r))


def main():
    guard_camera()
    go = "--go" in sys.argv
    do_idx = opt("do", 1)
    plan = json.load(open(os.path.join(HERE, "cell_plan.json")))
    T = a_to_b_transform()

    cam = Camera()
    try:
        img, g, cls = find_grid_robust(cam)
    finally:
        cam.close()
    board = [("O" if k == "O" else "X" if k == "X" else " ") for k, _, _ in cls]
    print(describe(board))
    if winner(board) or " " not in board:
        print("已結束：", winner(board) or "和局")
        return
    me = whose_turn(board, opt("first", "X"))
    T_ = T
    # 只選「有手臂夠得到」且「另一臂可停到淨空 ≥ 門檻」的格；格 7 等邊界格需加 --allow-tight 才會選
    reachable = set()
    for i, v in plan.items():
        if not v["arm"] or int(i) in EXCLUDE_CELLS or in_keepout(v["arm"], v[v["arm"]]):
            continue
        o = "B" if v["arm"] == "A" else "A"
        if clearance(v["arm"], v[v["arm"]], T_, best_park(v["arm"], v[v["arm"]], T_)) >= CLEAR_MM or "--allow-tight" in sys.argv:
            reachable.add(int(i))
    print("可落子格（可達且淨空足夠）：%s" % sorted(reachable))
    forced = opt("cell", -1)
    if forced >= 0:
        cell, score = forced, None
        if board[cell] != " ":
            raise SystemExit("格 %d 已有棋子" % cell)
    else:
        cell, score = best_move(board, me, allowed=reachable)
        full, _ = best_move(board, me)
        if full != cell:
            print("⚠ 最佳格 %d 沒有手臂夠得到（近格自身干涉），改下 %d" % (full, cell))
    active = plan[str(cell)]["arm"]
    other = "B" if active == "A" else "A"
    txy = plan[str(cell)][active]
    PARK[other] = best_park(active, txy, T)
    c = clearance(active, txy, T)
    print("輪到 %s → 格 %d（分數 %s），由 %s 執行，目標 %s；另一臂 %s 停在 %s；淨空 %.0f mm（門檻 %.0f）"
          % (me, cell, score, active, np.round(txy, 1), other, PARK[other], c, CLEAR_MM))
    if in_keepout(active, txy) or not path_clear(active, PARK[active], txy):
        raise SystemExit("目標或路徑進入相機支架禁入區，不執行")
    if c < CLEAR_MM and "--allow-tight" not in sys.argv:
        raise SystemExit("碰撞風險：淨空不足，不執行（確認現場後可加 --allow-tight）")
    if not go:
        print("（乾跑，不動）")
        return

    arms = {n: connect_arm(n) for n in "AB"}
    try:
        for a in arms.values():
            a.wait_feed()
        # 1) 另一臂先到停放點（一次只動一台）
        o = arms[other]
        if not at(o, PARK[other]):
            enable(o, speed=15)
            go_park(o)
            print("%s 已停放 %s" % (other, o.pose().round(1)))
        o.dash.DisableRobot()
        # 2) 執行臂：懸停 → 下降 → DO 開/關 → 上升 → 回停放
        a = arms[active]
        enable(a, speed=15)
        if not at(a, PARK[active]):
            go_park(a)                           # 路線只從停放點出發做過檢查
        r = float(a.pose()[3])
        approach(a, txy)                         # 小半徑轉向 → 直線伸出（避開左右支架）
        print("%s 懸停於格 %d 上方 %s" % (active, cell, a.pose().round(1)))
        sync_move(a, "MovL(%f,%f,%f,%f)" % (txy[0], txy[1], Z_PLACE, r))
        print("%s 下降到放置高度 %s" % (active, a.pose().round(1)))
        print("DO(%d) 開（吸）→ %s" % (do_idx, a.dash.sendRecvMsg("DOExecute(%d,1)" % do_idx).strip()))
        time.sleep(0.5)
        print("DO(%d) 關（放）→ %s" % (do_idx, a.dash.sendRecvMsg("DOExecute(%d,0)" % do_idx).strip()))
        sync_move(a, "MovL(%f,%f,%f,%f)" % (txy[0], txy[1], Z_HOVER, r))
        go_park(a)
        print("%s 回停放 %s；落子流程完成 ✓（真空未接，請手動把 %s 放到格 %d）" % (active, a.pose().round(1), me, cell))
    except Exception as e:
        print("停止：", e)
        for x in arms.values():
            try:
                x.dash.ResetRobot()
            except Exception:
                pass
    finally:
        for x in arms.values():
            try:
                x.dash.sendRecvMsg("DOExecute(%d,0)" % do_idx) if x.name == active else None
                x.dash.DisableRobot()
            except Exception:
                pass
            x.close()


if __name__ == "__main__":
    main()

"""B 臂吸取／放置（真空 DO1）。所有路線走「直線縮回 R_IN → 小半徑轉向 → 直線伸出」，每段先查支架禁入區。

高度（2026-11-14 實測）：B 吸盤碰到棋盤上棋子頂面時 Z = -154.82；棋子厚 5 mm；料區桌面比棋盤低 13～14 mm（stock_B.json）。
    python pickplace.py pick-cell 4 --to=O        # 從格 4 吸起，放到 O 疊（疊上第 n 顆自動算高度）
    python pickplace.py --dry pick-cell 4 --to=O  # 只檢查不動
"""
import json
import os
import sys
import time

import numpy as np

from common import connect_arm, enable, sync_move
from plan_cells import hits_post
from play import PARK, R_IN, Z_HOVER
from world import load

HERE = os.path.dirname(os.path.abspath(__file__))
STOCK = json.load(open(os.path.join(HERE, "stock_B.json"), encoding="utf-8"))
Z_TOUCH_BOARD = STOCK["z_touch_board_B"]      # 棋盤上棋子頂面
PIECE = STOCK["piece_mm"]
SUCK_PRESS = float(next((a[8:] for a in sys.argv if a.startswith("--press=")), "1.0"))   # 吸取時多壓的 mm
COUNT_FILE = os.path.join(HERE, "stock_count.json")


def counts():
    try:
        return json.load(open(COUNT_FILE, encoding="utf-8"))
    except OSError:
        return {"X疊": 0, "O疊": 0}


def seg_clear(cal, pts, z):
    for q in pts:
        if hits_post(cal, q, z):
            return False
    return True


def travel_points(p0, p1, n=25):
    """p0 → 直線縮回 (R_IN, p0 方向) → 小半徑轉向 → 直線伸出到 p1。"""
    a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
    r0, r1 = np.hypot(*p0), np.hypot(*p1)
    pts = [((1 - t) * r0 + t * R_IN) * np.array([np.cos(a0), np.sin(a0)]) for t in np.linspace(0, 1, n)]
    pts += [R_IN * np.array([np.cos((1 - t) * a0 + t * a1), np.sin((1 - t) * a0 + t * a1)]) for t in np.linspace(0, 1, n)]
    pts += [((1 - t) * R_IN + t * r1) * np.array([np.cos(a1), np.sin(a1)]) for t in np.linspace(0, 1, n)]
    return pts


def travel(arm, p1, r):
    """在 Z_HOVER 從目前位置移到 p1（三段）。"""
    p0 = arm.pose()[:2]
    a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
    if np.hypot(*p0) > R_IN + 2:
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (R_IN * np.cos(a0), R_IN * np.sin(a0), Z_HOVER, r))
    sync_move(arm, "MovJ(%f,%f,%f,%f)" % (R_IN * np.cos(a1), R_IN * np.sin(a1), Z_HOVER, r))
    sync_move(arm, "MovL(%f,%f,%f,%f)" % (p1[0], p1[1], Z_HOVER, r))


def main():
    dry = "--dry" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cell = int(args[1])
    to = next(a[5:] for a in sys.argv if a.startswith("--to=")) + "疊"
    plan = json.load(open(os.path.join(HERE, "cell_plan.json"), encoding="utf-8"))
    cal = load("B")
    src = np.array(plan[str(cell)]["B"])
    dst = np.array(STOCK["stocks"][to]["xy"])
    # 可直接指定棋子實際中心（--src=x,y，B 座標）與吸取偏移（--off=dx,dy）：吸盤要落在平坦面上（綠十字或黑底內，不可跨邊界）
    srcarg = next((a[6:] for a in sys.argv if a.startswith("--src=")), None)
    offarg = next((a[6:] for a in sys.argv if a.startswith("--off=")), None)
    if srcarg:
        src = np.array([float(v) for v in srcarg.split(",")])
    manual_off = np.array([float(v) for v in offarg.split(",")]) if offarg else None
    n = counts()[to]
    z_pick = Z_TOUCH_BOARD - SUCK_PRESS
    z_place = STOCK["stocks"][to]["z_first"] + n * PIECE
    print("B：格 %d %s 吸取 Z=%.1f → %s %s 第 %d 顆 放置 Z=%.1f" % (cell, src, z_pick, to, dst, n + 1, z_place))

    arm = connect_arm("B")
    try:
        arm.wait_feed()
        r = float(arm.pose()[3])
        cur = arm.pose()[:2]
        if manual_off is not None:
            print("指定吸取偏移 %s（%.1f mm）；放置點同步偏移" % (manual_off.round(1), np.hypot(*manual_off)))
            src, dst = src + manual_off, dst + manual_off
        elif "--offset=auto" in sys.argv:
            # 棋子直徑 80 mm：吸盤不必對正中央。中心在吸取高度 IK 不可達時，沿徑向（太遠往內、太近往外）
            # 找最小偏移（≤ 25 mm）；放置時加同樣偏移，讓棋子中心落在疊的正上方。
            ok = lambda xy, zs: all(arm.dash.InverseSolution(xy[0], xy[1], z, r, 0, 0).startswith("0,") for z in zs)
            u = src / np.hypot(*src)
            off = None
            for d in np.arange(0, 25.5, 1.0):
                for sgn in (-1, 1):
                    cand = src + sgn * d * u
                    if ok(cand, (z_pick, Z_HOVER)) and ok(dst + (cand - src), (z_place, Z_HOVER)):
                        off = cand - src
                        break
                if off is not None:
                    break
            if off is None:
                raise SystemExit("偏移 25 mm 內都找不到可吸取的點")
            print("吸取點偏移 %s（%.0f mm）；放置點同步偏移" % (off.round(1), np.hypot(*off)))
            src, dst = src + off, dst + off
        legs = [("到格 %d" % cell, cur, src), ("到%s" % to, src, dst), ("回停放", dst, np.array(PARK["B"][:2]))]
        for name, a, b in legs:
            if not seg_clear(cal, travel_points(a, b), Z_HOVER):
                raise SystemExit("路線「%s」會碰到支架，停止" % name)
        for xy, zs, what in ((src, (z_pick, Z_HOVER), "吸取點"), (dst, (z_place, Z_HOVER), "放置點")):
            for z in zs:
                if not arm.dash.InverseSolution(xy[0], xy[1], z, r, 0, 0).startswith("0,"):
                    raise SystemExit("%s IK 不可達：%s Z=%.1f（半徑 %.0f mm）" % (what, xy, z, np.hypot(*xy)))
        if hits_post(cal, src, z_pick) or hits_post(cal, dst, z_place):
            raise SystemExit("吸取／放置點會碰到支架")
        print("檢查通過（路線、IK、支架）")
        if dry:
            return
        enable(arm, speed=10)
        p = arm.pose()
        at_src = np.abs(p[:2] - src).max() < 1.5
        if not at_src:
            if p[2] < Z_HOVER - 1:
                sync_move(arm, "MovL(%f,%f,%f,%f)" % (p[0], p[1], Z_HOVER, r))
            travel(arm, src, r)
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (src[0], src[1], z_pick, r))
        print("  吸：", arm.dash.sendRecvMsg("DOExecute(1,1)").strip())
        time.sleep(0.8)
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (src[0], src[1], Z_HOVER, r))
        print("  已抬到 Z=%.0f，確認有吸起…" % Z_HOVER)
        # 吸起確認：手臂移開（縮回 R_IN）後看該格是否已空；棋子還在就放棄，不搬空手、不加計數
        a0 = np.arctan2(src[1], src[0])
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (R_IN * np.cos(a0), R_IN * np.sin(a0), Z_HOVER, r))
        from board import classify_cells, find_grid
        from common import Camera
        cam = Camera()
        try:
            img = cam.grab(skip=3, timeout=20)
        finally:
            cam.close()
        g = find_grid(img)
        still = classify_cells(img, g)[cell][0]
        if still != "empty":
            print("  ✗ 格 %d 仍有 %s：沒吸住。關閉真空、回停放，不加計數" % (cell, still))
            arm.dash.sendRecvMsg("DOExecute(1,0)")
            travel(arm, np.array(PARK["B"][:2]), r)
            raise SystemExit("沒吸住")
        print("  ✓ 格 %d 已空，棋子在吸盤上" % cell)
        travel(arm, dst, r)
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (dst[0], dst[1], z_place, r))
        print("  放：", arm.dash.sendRecvMsg("DOExecute(1,0)").strip())
        time.sleep(0.6)
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (dst[0], dst[1], Z_HOVER, r))
        c = counts()
        c[to] += 1
        json.dump(c, open(COUNT_FILE, "w", encoding="utf-8"), ensure_ascii=False)
        travel(arm, np.array(PARK["B"][:2]), r)
        print("完成：%s 現有 %d 顆；B 回停放 %s" % (to, c[to], arm.pose().round(1)))
    except BaseException as e:
        print("停止：", e)
        try:
            arm.dash.ResetRobot()
        except Exception:
            pass
        raise
    finally:
        try:
            arm.dash.DisableRobot()
        except Exception:
            pass
        arm.close()


if __name__ == "__main__":
    main()

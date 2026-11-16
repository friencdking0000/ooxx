"""依目前的手眼校正重算每格的指派（不動手臂）：控制器 IK（懸停／放置高度）＋支架禁入區＋兩臂淨空。

支架禁入區（2026-11-14）：左右兩根鋁擠橫桿在相機高度附近，A 在 (230~310, 120, -40) 時擦到左側那根。
判斷用「方盒頂面在該高度的影像位置」（不是 TCP 投影到棋盤面），方盒半寬 BOX_HALF_PX 當餘裕。
    python plan_cells.py          # 寫 cell_plan.json（play.py／selfplay.py 讀）
"""
import json
import os

import cv2
import numpy as np

from board import find_grid
from common import connect_arm
from world import arm_to_px, load, px_to_arm

HERE = os.path.dirname(os.path.abspath(__file__))
Z_HOVER, Z_PLACE = -40.0, -90.0
# 影像座標（全解析度 1280x720），支架在相機高度附近的範圍；擦撞時方盒頂面約在 x≈340、y 320–525
POSTS = {"左側支架": (0, 330, 380, 470), "右側支架": (990, 360, 1280, 480)}
BOX_HALF_PX = 75.0


def face_px(cal, xy, z, j1_deg=None):
    """手臂 TCP 在 (xy, z) 時，方盒頂面中心在影像的位置。"""
    th = np.arctan2(xy[1], xy[0]) if j1_deg is None else np.radians(j1_deg)
    o = np.array(cal["o"])
    f = np.asarray(xy, float) + [np.cos(th) * o[0] - np.sin(th) * o[1], np.sin(th) * o[0] + np.cos(th) * o[1]]
    return arm_to_px(cal, f, depth=cal["D_ref"] + (cal["z0"] - z))


# 2026-11-15 用深度量過：畫面右側那一段（托架＋橫桿）頂面離相機 514～519 mm，只比桌面高約 35 mm（它是腳座，不是直立的柱子）。
# 所以右側：手臂在懸停高度從上面過去不會碰到；只有往下降到接近桌面時，吸盤方盒不能壓到它。
# 判斷改在「桌面這個平面」做（方盒頂面的影像位置有透視放大，會把離支架還有 9 cm 的料疊也擋掉——使用者：明明夾得到）。
LOW_POSTS = {"右側支架"}
LOW_POST_DEPTH = 516.0        # mm：右側托架頂面離相機
LOW_POST_Z = -110.0           # 手臂 Z 低於這個高度才需要檢查右側（Z=-110 時吸盤底約高出桌面 60 mm）
LOW_POST_MARGIN_PX = 48.0     # 方盒半寬 22 mm＋餘裕 15 mm（桌面約 0.78 mm/px）


def hits_post(cal, xy, z):
    p = face_px(cal, xy, z)
    for name, (x0, y0, x1, y1) in POSTS.items():
        if name in LOW_POSTS:
            if z > LOW_POST_Z:
                continue
            q = arm_to_px(cal, np.asarray(xy, float), depth=LOW_POST_DEPTH)
            if x0 - LOW_POST_MARGIN_PX <= q[0] <= x1 + LOW_POST_MARGIN_PX and y0 - LOW_POST_MARGIN_PX <= q[1] <= y1 + LOW_POST_MARGIN_PX:
                return name
            continue
        if x0 - BOX_HALF_PX <= p[0] <= x1 + BOX_HALF_PX and y0 - BOX_HALF_PX <= p[1] <= y1 + BOX_HALF_PX:
            return name
    return None


def path_xy(p0, p1, n=40):
    r0, r1 = np.hypot(*p0[:2]), np.hypot(*p1[:2])
    a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
    return [((1 - t) * r0 + t * r1) * np.array([np.cos((1 - t) * a0 + t * a1), np.sin((1 - t) * a0 + t * a1)])
            for t in np.linspace(0, 1, n)]


def path_hits(cal, p0, p1, z=Z_HOVER):
    for q in path_xy(p0, p1):
        h = hits_post(cal, q, z)
        if h:
            return h
    return None


if __name__ == "__main__":
    from common import Camera
    from play import PARK
    cam = Camera()
    try:
        img = cam.grab(skip=5, timeout=20)
    finally:
        cam.close()
    g = find_grid(img)
    cals = {n: load(n) for n in "AB"}
    arms = {n: connect_arm(n) for n in "AB"}
    for a in arms.values():
        a.wait_feed()
    plan = {}
    print("格  影像中心     A：座標 / IK / 支架        B：座標 / IK / 支架        → 指派")
    for i, p in enumerate(g["cells"]):
        row = {"px": [float(p[0]), float(p[1])]}
        info = {}
        for n in "AB":
            xy = px_to_arm(cals[n], p)
            r = float(arms[n].pose()[3])
            ik = all(arms[n].dash.InverseSolution(xy[0], xy[1], z, r, 0, 0).startswith("0,") for z in (Z_HOVER, Z_PLACE))
            from play import path_clear
            post = (hits_post(cals[n], xy, Z_HOVER) or hits_post(cals[n], xy, Z_PLACE)
                    or (None if path_clear(n, PARK[n], xy) else "支架（路線）"))
            row[n] = xy.round(1).tolist()
            info[n] = (ik, post, float(np.hypot(*xy)))
        from play import CLEAR_MM, a_to_b_transform, clearance
        T = a_to_b_transform()
        for n in "AB":
            other = "B" if n == "A" else "A"
            cl = clearance(n, row[n], T, PARK[other])
            if info[n][0] and not info[n][1] and cl < CLEAR_MM:
                info[n] = (info[n][0], "另一臂（淨空 %.0f mm）" % cl, info[n][2])
        ok = [n for n in "AB" if info[n][0] and not info[n][1]]
        row["arm"] = max(ok, key=lambda n: info[n][2]) if ok else None      # 兩台都行：選半徑較大（離近端極限遠）
        row["why_not"] = {n: ("IK 不可達" if not info[n][0] else "碰%s" % info[n][1]) for n in "AB" if n not in ok}
        plan[str(i)] = row
        print("%d  (%4.0f,%4.0f)  %-28s %-28s → %s" % (
            i, p[0], p[1],
            "%s %s %s" % (np.round(row["A"], 0), "OK" if info["A"][0] else "✗", info["A"][1] or "-"),
            "%s %s %s" % (np.round(row["B"], 0), "OK" if info["B"][0] else "✗", info["B"][1] or "-"),
            row["arm"] or "無"))
    for n in "AB":
        print("停放點 %s %s：IK %s，支架 %s" % (n, PARK[n], arms[n].dash.InverseSolution(*PARK[n], float(arms[n].pose()[3]), 0, 0).split(",")[0],
                                         hits_post(cals[n], PARK[n][:2], PARK[n][2]) or "-"))
        arms[n].close()
    json.dump(plan, open(os.path.join(HERE, "cell_plan.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("已寫 cell_plan.json")

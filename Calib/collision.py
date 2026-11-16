"""兩臂碰撞預檢（俯視、保守簡化）。手眼校正完成前用「影像推得的假設座標系」，校正後改用 calib_*.json。

模型：手臂 = 底座→TCP 線段（半徑 ARM_R，涵蓋大臂/小臂寬度）+ TCP 處夾爪圓盤（半徑 GRIP_R）。
      兩台各自的 (線段 + 圓盤) 之間最小距離 < SAFETY 視為有碰撞風險。
座標假設（由 B 的 J1 轉動探測與底座位置推得，見 CLAUDE.md）：
  B：J1 軸在影像 (600, 15)，+X 朝影像下方，+Y 朝影像右方
  A：J1 軸在影像 (575, 680)，+X 朝影像上方，+Y 朝影像左方
  比例 0.776 mm/px（棋盤面深度 534 / fx 688）
    python collision.py
"""
import numpy as np

S = 0.776
BASE_PX = {"A": np.array([575.0, 680.0]), "B": np.array([600.0, 15.0])}
AXES = {"A": (np.array([0.0, -1.0]), np.array([-1.0, 0.0])),     # (+X 方向, +Y 方向) 的影像單位向量
        "B": (np.array([0.0, 1.0]), np.array([1.0, 0.0]))}
ARM_R, GRIP_R, SAFETY = 60.0, 110.0, 40.0    # mm


def to_px(arm, x, y):
    ex, ey = AXES[arm]
    return BASE_PX[arm] + (x * ex + y * ey) / S


def to_arm(arm, px):
    ex, ey = AXES[arm]
    d = (np.asarray(px, float) - BASE_PX[arm]) * S
    return float(d @ ex), float(d @ ey)


def seg_dist(a0, a1, b0, b1):
    """兩線段最小距離（mm 座標）。"""
    best = 1e9
    for p, (q0, q1) in ((a0, (b0, b1)), (a1, (b0, b1)), (b0, (a0, a1)), (b1, (a0, a1))):
        d = q1 - q0
        t = np.clip(((p - q0) @ d) / (d @ d + 1e-9), 0, 1)
        best = min(best, float(np.linalg.norm(p - (q0 + t * d))))
    return best


def clearance(tcp_px_A, tcp_px_B):
    """兩臂（影像像素 TCP 位置）之間的淨空 mm：線段距離 − (ARM_R*2)，與夾爪圓盤距離 − (GRIP_R*2) 取小。"""
    a0, a1 = BASE_PX["A"] * S, np.asarray(tcp_px_A, float) * S
    b0, b1 = BASE_PX["B"] * S, np.asarray(tcp_px_B, float) * S
    link = seg_dist(a0, a1, b0, b1) - 2 * ARM_R
    grip = float(np.linalg.norm(a1 - b1)) - 2 * GRIP_R
    return min(link, grip)


# 目前停放姿態（2026-11-10 實測的 XY，手臂座標）：A (83.5,170.5)、B (128.5,-209.2)
PARK = {"A": to_px("A", 83.5, 170.5), "B": to_px("B", 128.5, -209.2)}

if __name__ == "__main__":
    import json
    from board import find_grid_robust
    from common import Camera
    from reach import assignment, cell_report

    cam = Camera()
    try:
        _, g, cls = find_grid_robust(cam)
    finally:
        cam.close()
    asg = assignment(cell_report(g["cells"]))
    print("停放位置推得的影像位置：A %s，B %s（假設座標系）" % (PARK["A"].round(0), PARK["B"].round(0)))
    print("格  指派  手臂目標 px          另一台停放時的淨空 mm    判定")
    worst = 1e9
    for i, p in enumerate(g["cells"]):
        a = asg[i]
        if not a:
            print("%d   --   沒有人夠得到" % i)
            continue
        other = "B" if a == "A" else "A"
        tcp = {a: p, other: PARK[other]}
        c = clearance(tcp["A"], tcp["B"])
        worst = min(worst, c)
        print("%d   %s     (%4.0f,%4.0f)          %8.0f             %s" % (
            i, a, p[0], p[1], c, "OK" if c >= SAFETY else "⚠ 風險，需改停放位"))
    print("最小淨空 %.0f mm（門檻 %.0f mm）" % (worst, SAFETY))

"""可達性／分工分析：每個格子哪台手臂夠得到（用 IK 掃出的徑向可達範圍 + 影像幾何先驗）。

    python reach.py                 # 讀目前相機圖 + reach_<arm>_radial.json，印 9 格可達表
先驗（待手眼校正取代）：像素→mm 比例 = 棋盤面深度 / fx；A、B 的 J1 軸位置由底座方盒的影像位置估計。
可達判斷：徑向距離落在 [r_min + margin, r_max - margin]。margin 吸收先驗誤差（預設 25 mm）。
"""
import json
import os

import numpy as np

from board import classify_cells, find_grid

HERE = os.path.dirname(os.path.abspath(__file__))
FX = 687.8
BOARD_DEPTH = 534.0                    # mm，實測棋盤面
S = BOARD_DEPTH / FX                   # mm / px ≈ 0.776
# J1 軸的影像位置（底座方盒中心；盒子被畫面截斷，所以是由盒寬推估）。校正後以 calib 取代。
BASE_PX = {"A": np.array([575.0, 680.0]), "B": np.array([600.0, 15.0])}
MARGIN = 25.0


def radial_limits(arm, z):
    """從 reach_<arm>_radial.json 內插某高度 z 的 (r_min, r_max)。"""
    tab = json.load(open(os.path.join(HERE, "reach_%s_radial.json" % "B"))) if arm else None
    zs = sorted(int(k) for k in tab)
    lo = np.interp(z, zs, [tab[str(k)][0] for k in zs])
    hi = np.interp(z, zs, [tab[str(k)][1] for k in zs])
    return lo, hi


def cell_report(cells_px, z=-100):
    rows = []
    for i, p in enumerate(cells_px):
        row = {"cell": i}
        for arm in ("A", "B"):
            r = float(np.linalg.norm(p - BASE_PX[arm]) * S)
            lo, hi = radial_limits(arm, z)
            row[arm] = (r, lo + MARGIN <= r <= hi - MARGIN, r - lo)
        rows.append(row)
    return rows


def assignment(rows):
    """每格選一台：兩台都夠就選距離下限較遠（離自身干涉較遠）的那台。"""
    out = {}
    for row in rows:
        ok = [a for a in ("A", "B") if row[a][1]]
        out[row["cell"]] = max(ok, key=lambda a: row[a][2]) if ok else None
    return out


if __name__ == "__main__":
    from common import Camera
    from board import find_grid_robust
    cam = Camera()
    try:
        img, g, cls = find_grid_robust(cam)
    finally:
        cam.close()
    rows = cell_report(g["cells"])
    asg = assignment(rows)
    print("比例 %.3f mm/px；margin %.0f mm；可達範圍用 B 的 IK 掃描（兩台同型號）" % (S, MARGIN))
    print("格  內容   A 半徑(可達)    B 半徑(可達)   指派")
    for row, (k, _, _) in zip(rows, cls):
        a, b = row["A"], row["B"]
        print("%d   %-5s  %6.0f (%s)      %6.0f (%s)     %s" % (
            row["cell"], k, a[0], "Y" if a[1] else "N", b[0], "Y" if b[1] else "N", asg[row["cell"]] or "沒有人夠得到"))

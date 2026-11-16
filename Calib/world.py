"""校正後的座標換算：影像像素 ↔ 各手臂 XY（棋盤面），以及兩臂之間的轉換。

    python world.py          # 交叉驗證 A／B 校正、印兩臂底座相對位置、各格在兩臂座標
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(arm):
    return json.load(open(os.path.join(HERE, "calib_%s.json" % arm), encoding="utf-8"))


def px_to_arm(cal, px, depth=None):
    """棋盤面（或指定深度 mm）上的像素 → 手臂 TCP XY。"""
    k = (depth or cal["D_board"]) / cal["fx"] / cal["s"]
    u, v = px[0] - cal["cx"], (px[1] - cal["cy"]) * cal["flip"]
    a, b = cal["a"] * k, cal["b"] * k
    return np.array([a * u - b * v + cal["tx"], b * u + a * v + cal["ty"]])


def arm_to_px(cal, xy, depth=None):
    """手臂 XY（位於棋盤面或指定深度）→ 影像像素。px_to_arm 的反函數。"""
    k = (depth or cal["D_board"]) / cal["fx"] / cal["s"]
    a, b = cal["a"] * k, cal["b"] * k
    d = np.asarray(xy, float) - [cal["tx"], cal["ty"]]
    det = a * a + b * b
    u = (a * d[0] + b * d[1]) / det
    v = (-b * d[0] + a * d[1]) / det
    return np.array([u + cal["cx"], v * cal["flip"] + cal["cy"]])


def rigid2d(P, Q):
    """Q ≈ R P + t（2D 剛體），回傳 (R, t, 殘差)。"""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    pm, qm = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pm).T @ (Q - qm))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    t = qm - R @ pm
    return R, t, np.linalg.norm((R @ P.T).T + t - Q, axis=1)


if __name__ == "__main__":
    import cv2
    from board import find_grid
    A, B = load("A"), load("B")
    img = cv2.imread(os.path.join(HERE, "out", "cb_B_home.png"))
    g = find_grid(img)
    pa = np.array([px_to_arm(A, p) for p in g["cells"]])
    pb = np.array([px_to_arm(B, p) for p in g["cells"]])
    R, t, err = rigid2d(pa, pb)
    print("交叉驗證：9 格在 A、B 座標之間的剛體擬合殘差 平均 %.2f mm、最大 %.2f mm" % (err.mean(), err.max()))
    print("A 的原點在 B 座標：(%.1f, %.1f)，距離 %.1f mm；A 的 +X 在 B 座標指向 %.1f°"
          % (t[0], t[1], np.linalg.norm(t), np.degrees(np.arctan2(R[1, 0], R[0, 0]))))
    print("底座在影像（棋盤面）：A %s、B %s" % (arm_to_px(A, (0, 0)).round(0), arm_to_px(B, (0, 0)).round(0)))
    print("\n格   A 座標 (X,Y)      半徑    B 座標 (X,Y)      半徑")
    for i in range(9):
        print("%d  (%6.1f,%6.1f) %5.0f   (%6.1f,%6.1f) %5.0f" % (i, *pa[i], np.hypot(*pa[i]), *pb[i], np.hypot(*pb[i])))

"""手眼校正：拖曳示教取點。你把夾爪中心（TCP）用手帶到棋盤上的地標，按 Enter，我記下手臂姿態。

    python teach_points.py B            # 互動：依序教 8 個地標（輸入 s 跳過、q 結束）
    python teach_points.py B --fit      # 用已存的點擬合 像素→手臂座標，印殘差

地標（都在木板表面、沒有圓片的位置）：4 個膠帶內部交點 + 4 個空格中心（邊中格 1,3,5,7）。
拖曳：手臂須在使能狀態，按住前臂上的拖曳按鈕移動（或 DobotStudio 的拖曳模式）。
結果存 teach_<arm>.json（像素座標、手臂 X Y Z R）。校正的 TCP = 你每次對準地標的那個夾爪點，請每次用同一個點。
"""
import json
import sys

import cv2
import numpy as np

from board import classify_cells, find_grid
from common import OUT, Camera, connect_arm


def landmarks(img):
    g = find_grid(img)
    names, pts = [], []
    for n, p in zip(("交點TL", "交點TR", "交點BL", "交點BR"), g["corners"]):
        names.append(n), pts.append(p)
    cls = classify_cells(img, g)
    for i in (1, 3, 5, 7):
        if cls[i][0] == "empty":
            names.append("格%d中心" % i), pts.append(g["cells"][i])
    return names, np.array(pts, float), g


def teach(name):
    cam = Camera()
    try:
        img = cam.grab(n=3)
    finally:
        cam.close()
    names, pts, g = landmarks(img)
    arm = connect_arm(name)
    arm.wait_feed()
    rows = []
    try:
        for n, p in zip(names, pts):
            vis = img.copy()
            cv2.circle(vis, (int(p[0]), int(p[1])), 12, (0, 0, 255), 3)
            cv2.imwrite("%s/teach_target.png" % OUT, vis)       # 目標標記，可開啟對照
            ans = input("[%s] 地標 %s 像素 (%.0f,%.0f)（見 out/teach_target.png）→ 夾爪中心對準後按 Enter，s 跳過，q 結束： "
                        % (name, n, p[0], p[1])).strip().lower()
            if ans == "q":
                break
            if ans == "s":
                continue
            pose = arm.pose().tolist()
            rows.append(dict(name=n, px=[float(p[0]), float(p[1])], pose=pose))
            print("  記錄 %s 姿態 X%.1f Y%.1f Z%.1f R%.1f" % ((n,) + tuple(pose)))
    finally:
        arm.close()
    json.dump(dict(arm=name, points=rows), open("%s/../teach_%s.json" % (OUT, name), "w"), indent=1)
    print("已存 teach_%s.json（%d 點）" % (name, len(rows)))


def fit(name):
    d = json.load(open("%s/../teach_%s.json" % (OUT, name)))["points"]
    px = np.array([r["px"] for r in d], np.float32)
    xy = np.array([r["pose"][:2] for r in d], np.float32)
    z = np.array([r["pose"][2] for r in d])
    if len(d) < 4:
        sys.exit("至少需要 4 點")
    H, _ = cv2.findHomography(px, xy, 0)
    pred = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(pred - xy, axis=1)
    print("%s：%d 點，殘差 平均 %.2f mm 最大 %.2f mm；板面 Z 平均 %.1f（標準差 %.1f）"
          % (name, len(d), err.mean(), err.max(), z.mean(), z.std()))
    json.dump(dict(arm=name, H=H.tolist(), board_z=float(z.mean())),
              open("%s/../calib_%s.json" % (OUT, name), "w"), indent=1)


if __name__ == "__main__":
    (fit if "--fit" in sys.argv else teach)(sys.argv[1])

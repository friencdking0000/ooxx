"""校正外推驗證：手臂到某格上方的 Z（校正沒用過的高度與位置），拍照找方盒頂面，與校正模型預測比較。

    python verify.py A 1 --z=-90 --go       # A 到格 1 上方 Z=-90（另一臂需已停放、不在路徑上）
誤差換算成 mm（以頂面所在高度的比例），並存疊圖 out/verify_<arm>_<cell>.png。
"""
import json
import os
import sys
import time

import cv2
import numpy as np

from calib_box import find_face
from common import OUT, Camera, connect_arm, enable, joint_angles, sync_move
from world import arm_to_px, load

HERE = os.path.dirname(os.path.abspath(__file__))
# 模板拍攝時頂面到相機距離：B 模板於 Z=0（244 mm）、A 模板於 Z=-40（281 mm）
TEMPLATE = {"A": (-40.0, 281.0), "B": (0.0, 244.0)}   # 模板拍攝時（舊相機高度）；實際比例在預測位置附近再掃描


def main():
    arm_name, cell = sys.argv[1], int(sys.argv[2])
    z = float(next((a[4:] for a in sys.argv if a.startswith("--z=")), "-90"))
    go = "--go" in sys.argv
    cal = load(arm_name)
    plan = json.load(open(os.path.join(HERE, "cell_plan.json")))
    if plan[str(cell)]["arm"] != arm_name and arm_name not in ("A", "B"):
        raise SystemExit("格 %d 不歸 %s" % (cell, arm_name))
    txy = np.array(plan[str(cell)][arm_name])
    z_t, d_t = TEMPLATE[arm_name]
    d_face = d_t + (z_t - z)                       # Z 越低，頂面離相機越遠
    print("%s → 格 %d %s，Z=%.0f；預期頂面距相機 %.0f mm" % (arm_name, cell, txy.round(1), z, d_face))
    if not go:
        return
    arm = connect_arm(arm_name)
    cam = Camera()
    try:
        arm.wait_feed()
        from play import PARK, approach, at, go_park, in_keepout, path_clear
        if in_keepout(arm_name, txy) or not path_clear(arm_name, PARK[arm_name], txy):
            raise RuntimeError("目標或路線會碰到支架，不驗證這格")
        enable(arm, speed=15)
        if not at(arm, PARK[arm_name]):
            go_park(arm)
        r = float(arm.pose()[3])
        approach(arm, txy)                       # 先在小半徑轉向，再直線伸出（避開支架）
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (txy[0], txy[1], z, r))
        time.sleep(0.5)
        img = cam.grab(skip=3, timeout=20)
        j1 = joint_angles(arm)[0]
        sync_move(arm, "MovL(%f,%f,%f,%f)" % (txy[0], txy[1], -40, r))
        go_park(arm)
        arm.dash.DisableRobot()
        th = np.radians(j1)
        o = np.array(cal["o"])
        face_xy = txy + np.array([np.cos(th) * o[0] - np.sin(th) * o[1], np.sin(th) * o[0] + np.cos(th) * o[1]])
        pred = arm_to_px(cal, face_xy, depth=cal["D_ref"] + (cal["z0"] - z))
        # 在預測位置 ±110 px 內、多個比例下找頂面（遠離光軸時透視變形大，全圖搜尋易錯配）
        win = (int(pred[0]) - 110, int(pred[1]) - 110, int(pred[0]) + 110, int(pred[1]) + 110)
        best = None
        for k in np.arange(0.85, 1.25, 0.03) * (d_t / d_face):
            r = find_face(img, j1=j1, scale=k, arm=arm_name, search=win)
            if best is None or r[2] > best[2]:
                best = r
        c, ang, sc = best
        err_px = float(np.linalg.norm(c - pred))
        mm_per_px = (cal["D_ref"] + (cal["z0"] - z)) / cal["fx"]
        print("頂面偵測 (%.1f,%.1f) 分數 %.3f 旋轉 %+.1f（J1 %+.1f）；模型預測 (%.1f,%.1f)；誤差 %.1f px ≈ %.1f mm"
              % (c[0], c[1], sc, ang, j1, pred[0], pred[1], err_px, err_px * mm_per_px))
        vis = img.copy()
        cv2.circle(vis, tuple(int(v) for v in pred), 10, (0, 255, 0), 3)
        cv2.circle(vis, tuple(int(v) for v in c), 6, (0, 0, 255), -1)
        cv2.imwrite("%s/verify_%s_%d.png" % (OUT, arm_name, cell), vis)
    except Exception as e:
        print("停止：", e)
        arm.dash.ResetRobot()
    finally:
        try:
            arm.dash.DisableRobot()
        except Exception:
            pass
        cam.close()
        arm.close()


if __name__ == "__main__":
    main()

"""懸停驗證：用手動教點算出的新座標（Calib/teach_fit.json），讓手臂停在指定格子正上方，給人看吸盤有沒有對準，並把結果記錄下來。

    python hover_check.py                  # 只印計畫（不動）
    python hover_check.py --go             # 會動：預設 B 格4、B 格6、A 格4、A 格2
    python hover_check.py --go B:4 A:8 --hold=10 --h=20

- 格號是畫面格號（0 左上 … 8 右下）。B 的「點 3」＝畫面格 6，A 的「點 3」＝畫面格 2（jog 頁各臂視角的編號）。
- 高度：用該臂四個交叉點記錄時的 Z 擬合一個平面，算出格子處的棋盤面，再往上 --h（預設 20 mm）。
  教點時吸盤是壓著的，所以吸盤口實際離棋盤約 15～20 mm。
- 開始前兩臂都先回停放點；一次只有一台在棋盤上方；只在空中停，真空關閉，不碰任何東西。
- 記錄：Arena/out/hover_check_<時間>.json（目標、實際姿態、關節角）＋ 每個點一張俯視照片
  （照片上畫了：紅圈＝舊相機校正認為的格子中心）。
"""
import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Calib"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "MG400"))

import cv2                                      # noqa: E402
import numpy as np                              # noqa: E402

import mover                                    # noqa: E402
from common import OrbbecCamera, enable, sync_move    # noqa: E402
from play import Z_HOVER                        # noqa: E402

FIT = os.path.join(os.path.dirname(HERE), "Calib", "teach_fit.json")
PLAN = os.path.join(os.path.dirname(HERE), "Calib", "cell_plan.json")
OUT = os.path.join(HERE, "out")
DEFAULT = ["B:4", "B:6", "A:4", "A:2"]


def board_plane(fit_arm):
    """四個交叉點的 (x, y, z) → z = a·x + b·y + c。"""
    pts = [(fit_arm["cells"][k][0], fit_arm["cells"][k][1], fit_arm["board_z"][k])
           for k in ("c0", "c1", "c2", "c3") if k in fit_arm["board_z"]]
    P = np.array(pts, float)
    coef, *_ = np.linalg.lstsq(np.c_[P[:, :2], np.ones(len(P))], P[:, 2], rcond=None)
    return lambda xy: float(coef @ [xy[0], xy[1], 1.0])


def main():
    go = "--go" in sys.argv
    hold = float(next((a[7:] for a in sys.argv if a.startswith("--hold=")), "10"))
    h = float(next((a[4:] for a in sys.argv if a.startswith("--h=")), "20"))
    targets = [a for a in sys.argv[1:] if ":" in a and not a.startswith("--")] or DEFAULT
    fit = json.load(open(FIT, encoding="utf-8"))
    plan = json.load(open(PLAN, encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    record = dict(time=stamp, hold_s=hold, above_board_mm=h, fit_file=FIT, points=[])

    jobs = []
    for t in targets:
        name, cell = t.split(":")
        xy = np.array(fit[name]["cells"][cell][:2], float)
        z = board_plane(fit[name])(xy) + h
        old = np.array(plan[cell][name][:2], float)
        jobs.append((name, cell, xy, z, old))
        print("%s 格%s：新座標 %s Z=%.1f（舊 cell_plan %s，差 %.1f mm）" % (
            name, cell, xy.round(1), z, old.round(1), np.hypot(*(xy - old))), flush=True)
    if not go:
        print("（只印計畫；加 --go 才會動）")
        return

    cam = None
    try:
        cam = OrbbecCamera(depth=False)
    except Exception as e:
        print("相機開不起來，這次不拍照：", e, flush=True)
    M = mover.Mover()
    try:
        for a in M.arms.values():
            enable(a.arm, speed=mover.TRAVEL_SPEED)
            a.arm.dash.sendRecvMsg("SetCollisionLevel(3)")
            a.arm.dash.sendRecvMsg("DOExecute(1,0)")
        for a in M.arms.values():
            a.go_park()
        for name, cell, xy, z, old in jobs:
            a = M.arms[name]
            if not (a.ik(xy, z) and a.ik(xy, Z_HOVER)):
                print("%s 格%s：IK 到不了，略過" % (name, cell), flush=True)
                continue
            if a.post(xy, z) or a.post(xy, Z_HOVER):
                print("%s 格%s：太靠近支架，略過" % (name, cell), flush=True)
                continue
            a.goto(xy)
            a.arm.dash.SpeedFactor(10)
            sync_move(a.arm, "MovL(%f,%f,%f,%f)" % (xy[0], xy[1], z, a.r))
            time.sleep(0.5)
            pose = [round(float(v), 2) for v in a.arm.pose()[:4]]
            rec = getattr(a.arm, "last_rec", None)
            joints = [round(float(v), 2) for v in rec["q_actual"][:4]] if rec is not None else None
            print(">>> %s 停在格%s 正上方（棋盤上約 %.0f mm），停 %.0f 秒；實際 %s 關節 %s" % (
                name, cell, h, hold, pose, joints), flush=True)
            photo = None
            if cam is not None:
                try:
                    img = cam.grab(skip=2, timeout=10)
                    px = tuple(int(round(v)) for v in plan[cell]["px"])
                    cv2.circle(img, px, 10, (0, 0, 255), 2)
                    cv2.putText(img, "%s cell%s  red=old calib center" % (name, cell), (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    photo = os.path.join(OUT, "hover_%s_%s_cell%s.jpg" % (stamp, name, cell))
                    cv2.imwrite(photo, img)
                except Exception as e:
                    print("   拍照失敗：", e, flush=True)
            record["points"].append(dict(arm=name, cell=cell, target_xy=xy.round(2).tolist(), target_z=round(z, 2),
                                         old_plan_xy=old.round(2).tolist(), diff_mm=round(float(np.hypot(*(xy - old))), 2),
                                         pose=pose, joints=joints, photo=photo,
                                         at=datetime.datetime.now().strftime("%H:%M:%S")))
            time.sleep(max(0.0, hold - 1.0))
            sync_move(a.arm, "MovL(%f,%f,%f,%f)" % (xy[0], xy[1], Z_HOVER, a.r))
            a.arm.dash.SpeedFactor(mover.TRAVEL_SPEED)
            a.go_park()
        print("完成，兩臂在停放點", flush=True)
    except BaseException as e:
        record["error"] = repr(e)
        print("停止：", repr(e), flush=True)
        M.park_all()
        raise
    finally:
        path = os.path.join(OUT, "hover_check_%s.json" % stamp)
        json.dump(record, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("已記錄 %s" % path, flush=True)
        if cam is not None:
            cam.close()
        M.close()


if __name__ == "__main__":
    main()

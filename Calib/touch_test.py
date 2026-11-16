"""接觸偵測實測（使用者：壓下去碰到的回饋可以當作有吸取到）：開真空 → 慢慢往下 → 看回饋判斷碰到 → 停 → 抬起。

    python touch_test.py <B x,y> <預估接觸 Z>      # 例：python touch_test.py 165.7,176.8 -155.5
記錄存 out/touch_log.csv（時間、Z 實際、Z 目標、落後、四軸電流、模式）。
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import connect_arm, enable, sync_move     # noqa: E402
from pickplace import travel                           # noqa: E402
from play import PARK, Z_HOVER                         # noqa: E402

LAG_MM = 3.0          # 實際 Z 落後目標超過這麼多＝被擋住（實測下降中本來就落後 ~1 mm，只當備援）
DI_J2 = 0.12          # J2 電流偏離「剛剛下降中的正常值」超過這麼多（A）＝碰到（2026-11-14 實測：碰到後 0.38 → 0.25 → -0.9）


def touch_down(a, xy, z_est, r, over=6.0, below=8.0, speed=5, log=None):
    """從 z_est+over 慢慢往 z_est-below 走，碰到就停。回傳 (接觸 Z 或 None, 原因)。"""
    sync_move(a, "MovL(%f,%f,%f,%f)" % (xy[0], xy[1], z_est + over, r))
    a.dash.sendRecvMsg("DOExecute(1,1)")
    time.sleep(0.3)
    a.dash.SpeedFactor(speed)
    z_floor = z_est - below
    import dual_sync
    dual_sync.check(a.move.sendRecvMsg("MovL(%f,%f,%f,%f)" % (xy[0], xy[1], z_floor, r)), "MovL down")
    t0 = time.perf_counter()
    reason, zc = None, None
    started = False
    hist = []                                 # 下降中 J2 電流（判斷基準）
    while time.perf_counter() - t0 < 15:
        rec = a.last_rec
        z = float(rec["tool_vector_actual"][2])
        zt = float(rec["Tool_vector_target"][2])
        mode = int(rec["robot_mode"])
        cur = [float(c) for c in rec["i_actual"][:4]]
        if log is not None:
            log.append([round(time.perf_counter() - t0, 3), round(z, 2), round(zt, 2), round(z - zt, 2), *[round(c, 3) for c in cur], mode])
        if mode == 7 or z < z_est + over - 0.5:
            started = True
        if mode == 9:
            reason, zc = "報警（碰撞偵測）", z
            break
        t = time.perf_counter() - t0
        if started and t > 0.25:
            if len(hist) >= 15 and abs(cur[1] - float(np.median(hist[-25:-5]))) > DI_J2:
                reason, zc = "J2 電流變化 %.2f A" % (cur[1] - float(np.median(hist[-25:-5]))), z
                break
            hist.append(cur[1])
        if started and z - zt > LAG_MM:
            reason, zc = "位置落後 %.1f mm" % (z - zt), z
            break
        if started and abs(z - z_floor) < 0.2 and mode == 5:
            reason = "到底都沒碰到"
            break
        time.sleep(0.005)
    a.dash.ResetRobot()                       # 停下動作（保持真空）
    time.sleep(0.2)
    if a.mode() == 9:
        a.dash.ClearError()
        time.sleep(0.3)
    if a.mode() != 5:
        a.dash.EnableRobot()
        time.sleep(1.0)
    a.dash.SpeedFactor(15)
    return zc, reason


def depth_top(cam, px):
    ds = [cam.grab(depth=True, skip=1, timeout=15)[1] for _ in range(5)]
    d = np.median(np.stack(ds), axis=0)
    u, v = int(px[0]), int(px[1])
    t = d[v - 12:v + 13, u - 12:u + 13]
    return float(np.median(t[t > 0]))


def main():
    from common import OrbbecCamera
    from world import arm_to_px, load
    xy = np.array([float(v) for v in sys.argv[1].split(",")])
    z_est = float(sys.argv[2])
    px = arm_to_px(load("B"), xy, depth=530)
    cam = OrbbecCamera(depth=True)
    a = connect_arm("B")
    log = []
    try:
        a.wait_feed()
        r = float(a.pose()[3])
        top0 = depth_top(cam, px)
        enable(a, speed=15)
        travel(a, xy, r)
        zc, why = touch_down(a, xy, z_est, r, log=log)
        print("接觸：%s，Z=%s（預估 %.1f）" % (why, None if zc is None else round(zc, 2), z_est))
        time.sleep(0.5)
        sync_move(a, "MovL(%f,%f,%f,%f)" % (xy[0], xy[1], Z_HOVER, r))
        a0 = np.arctan2(xy[1], xy[0])
        sync_move(a, "MovL(%f,%f,%f,%f)" % (215 * np.cos(a0), 215 * np.sin(a0), Z_HOVER, r))
        top1 = depth_top(cam, px)
        picked = top1 - top0 > 2.0
        print("疊頂深度 %.1f → %.1f mm（%+.1f）→ %s" % (top0, top1, top1 - top0, "✓ 吸起一顆" if picked else "✗ 沒吸起"))
        travel(a, xy, r)
        z_put = (zc if zc is not None else z_est) + 2.0
        sync_move(a, "MovL(%f,%f,%f,%f)" % (xy[0], xy[1], z_put, r))
        a.dash.sendRecvMsg("DOExecute(1,0)")
        time.sleep(0.6)
        sync_move(a, "MovL(%f,%f,%f,%f)" % (xy[0], xy[1], Z_HOVER, r))
        travel(a, np.array(PARK["B"][:2]), r)
        print("放回、B 回停放；疊頂深度 %.1f mm" % depth_top(cam, px))
    except BaseException as e:
        print("停止：", e)
        a.dash.sendRecvMsg("DOExecute(1,0)")
        a.dash.ResetRobot()
        raise
    finally:
        a.dash.DisableRobot()
        a.close()
        cam.close()
        with open(os.path.join(HERE, "out", "touch_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "z", "z_target", "lag", "i1", "i2", "i3", "i4", "mode"])
            w.writerows(log)


if __name__ == "__main__":
    main()

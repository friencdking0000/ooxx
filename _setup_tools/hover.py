"""兩臂先回停放點 → 找料區每一疊 → 由搆得到的那台移到疊的正上方（疊頂上方 15 mm）停幾秒給人看 → 回停放點。
python hover.py [--go] [--hold=6]   不加 --go 只印計畫、手臂不動。真空全程不開。"""
import sys, os, time, json
sys.stdout.reconfigure(encoding="utf-8")
R = os.path.join(os.environ["USERPROFILE"], "Desktop", "OOXX-robot-v1.1")
for d in ("Arena", "Calib", "MG400"): sys.path.insert(0, os.path.join(R, d))
os.chdir(os.path.join(R, "Arena"))
import numpy as np
import mover, stocks, align
from common import OrbbecCamera, enable, sync_move
from world import px_to_arm
from play import Z_HOVER
go = "--go" in sys.argv
hold = float(next((a[7:] for a in sys.argv if a.startswith("--hold=")), "6"))
plan = json.load(open(os.path.join(R, "Calib", "cell_plan.json"), encoding="utf-8"))
M0 = align.load()
cam = OrbbecCamera(depth=True, hires=True)
try:
    img = cam.grab(skip=3, timeout=15)
    ds = [cam.grab(depth=True, skip=1, timeout=15)[1] for _ in range(5)]
finally:
    cam.close()
img = align.apply(M0, img)
d = align.apply(M0, np.median(np.stack(ds), axis=0).astype(np.float32), depth=True)
M = mover.Mover()
try:
    B, A = M.arms["B"], M.arms["A"]
    S = [s for s in stocks.detect(img, d, B.cal, [plan[str(i)]["px"] for i in range(9)]) if s["kind"] in ("O", "X") and -175 < s["z_touch"] < -110]
    print("料區：", [(s["kind"], s["n"], tuple(int(t) for t in s["px"])) for s in S], flush=True)
    jobs = []
    for s in S:
        chosen = None
        for arm in (B, A):
            xy = np.asarray(s["xy"], float) if arm.name == "B" else np.asarray(px_to_arm(arm.cal, s["px"], depth=s["top"]), float)
            z = s["z_touch"] + arm.dz + 15.0
            why = []
            if not (arm.ik(xy, z) and arm.ik(xy, Z_HOVER)): why.append("IK 不可達")
            if arm.post(xy, z) or arm.post(xy, Z_HOVER): why.append("太靠近相機支架")
            if not arm.leg_ok(arm.park, xy): why.append("沒有不碰支架的路線")
            if not M.clear_of_other(arm, [np.asarray(s["xy"], float)]): why.append("離另一台太近")
            print("  %s 疊（影像 %s）→ %s 臂 座標 %s Z=%.1f：%s" % (s["kind"], tuple(int(t) for t in s["px"]), arm.name, xy.round(1), z, "可以" if not why else "、".join(why)), flush=True)
            if not why:
                chosen = (arm, xy, z)
                break
        if chosen:
            jobs.append((s, chosen))
    if not go:
        print("（乾跑，手臂不動）"); sys.exit(0)
    for arm in (B, A):                                   # 先回標準位置（停放點）
        enable(arm.arm, speed=mover.TRAVEL_SPEED)
        arm.arm.dash.sendRecvMsg("SetCollisionLevel(3)")
        arm.arm.dash.sendRecvMsg("DOExecute(1,0)")
        arm.go_park()
        print("%s 已到停放點 %s" % (arm.name, np.round(arm.arm.pose()[:3], 1)), flush=True)
    for s, (arm, xy, z) in jobs:
        arm.goto(xy)
        arm.arm.dash.SpeedFactor(10)
        sync_move(arm.arm, "MovL(%f,%f,%f,%f)" % (xy[0], xy[1], z, arm.r))
        print(">>> %s 臂現在在 %s 疊正上方（疊頂上方 15 mm），停 %.0f 秒" % (arm.name, s["kind"], hold), flush=True)
        time.sleep(hold)
        sync_move(arm.arm, "MovL(%f,%f,%f,%f)" % (xy[0], xy[1], Z_HOVER, arm.r))
        arm.arm.dash.SpeedFactor(mover.TRAVEL_SPEED)
        arm.go_park()
    for arm in (B, A):
        arm.arm.dash.DisableRobot()
    print("完成，兩臂都在停放點", flush=True)
except BaseException as e:
    print("停止：", repr(e), flush=True)
    M.park_all()
    raise
finally:
    M.close()

import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "Calib"))
import mover, find_pieces as fp
from a_grip_test_util import shot
from common import OrbbecCamera, enable, sync_move
from world import px_to_arm
plan = json.load(open(os.path.join(os.path.dirname(HERE), "Calib", "cell_plan.json"), encoding="utf-8"))
cam = OrbbecCamera(depth=True); img = cam.grab(skip=3, timeout=15); cam.close()
pcs = [p for p in fp.pieces(img, plan) if p["cell"] == 4]
if not pcs: raise SystemExit("格 4 沒有棋子")
M = mover.Mover()
try:
    A = M.arms["A"]
    c = px_to_arm(A.cal, pcs[0]["px"], depth=fp.PIECE_TOP)
    z_top = mover.Z_TOUCH_B + mover.board_dz(4) + 0.3 + A.dz; z_surf = mover.Z_TOUCH_B - 3.7 + mover.board_dz(4) + A.dz
    print("格 4 的 %s → A 座標 %s（半徑 %.0f），路線 %s" % (pcs[0]["kind"], c.round(1), np.hypot(*c), A.route_rin(A.park, c)), flush=True)
    if A.route_rin(A.park, c) is None or A.post(c, -40) or not A.ik(c, z_top - 10): raise SystemExit("A 到不了格 4")
    enable(A.arm, speed=15); A.arm.dash.sendRecvMsg("SetCollisionLevel(3)")
    A.goto(c)
    zc = A.touch(c, z_top, vacuum_on=True, below=9.0, z_surf=z_surf)
    print("接觸", zc, flush=True)
    if zc is not None:
        time.sleep(0.5); q = A.arm.pose()
        sync_move(A.arm, "MovL(%f,%f,%f,%f)" % (q[0], q[1], q[2] + 40.0, A.r)); time.sleep(1.0)
        shot("grip_cell4")
        sync_move(A.arm, "MovL(%f,%f,%f,%f)" % (q[0], q[1], zc + 2.0, A.r))
        A.arm.dash.sendRecvMsg("DOExecute(1,0)"); time.sleep(0.8)
    A.lift(); A.go_park()
except BaseException as e:
    print("停止：", repr(e), flush=True); M.park_all(); raise
finally:
    M.close()

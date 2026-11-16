import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
R = os.path.join(os.environ["USERPROFILE"], "Desktop", "OOXX-robot-v1.1")
for d in ("Arena", "Calib", "MG400"): sys.path.insert(0, os.path.join(R, d))
os.chdir(os.path.join(R, "MG400"))
from common import connect_arm, enable, sync_move
name = sys.argv[1]
a = connect_arm(name)
try:
    a.wait_feed()
    x, y, z, r = [float(v) for v in a.pose()[:4]]
    print("%s（IP %s）目前位置 X=%.1f Y=%.1f Z=%.1f" % (name, a.ip, x, y, z), flush=True)
    if z < -45:
        raise SystemExit("太低，不動")
    enable(a, speed=20)
    a.dash.sendRecvMsg("DOExecute(1,0)")
    for i in range(2):
        sync_move(a, "MovL(%f,%f,%f,%f)" % (x, y, z + 30, r)); time.sleep(0.4)
        sync_move(a, "MovL(%f,%f,%f,%f)" % (x, y, z, r)); time.sleep(0.4)
    print("%s 上下動了兩次，回到原位" % name, flush=True)
finally:
    try:
        a.dash.DisableRobot()
    except Exception:
        pass
    a.close()

import os, sys, time
R = os.path.join(os.environ["USERPROFILE"], "Desktop", "OOXX-robot-v1.1")
for d in ("Arena", "Calib", "MG400"):
    sys.path.insert(0, os.path.join(R, d))
os.chdir(os.path.join(R, "Arena"))
import mover
from common import sync_move
from play import Z_HOVER, a_to_b_transform
A = mover.Arm("A", a_to_b_transform())
a, d = A.arm, A.arm.dash
d.sendRecvMsg("DOExecute(1,0)")
d.sendRecvMsg("SetCollisionLevel(0)")
d.DisableRobot(); time.sleep(0.5)
d.ClearError(); time.sleep(0.5)
print("enable:", d.EnableRobot()); time.sleep(1.5)
print("mode", a.mode())
d.SpeedFactor(10)
q = a.pose()
sync_move(a, "MovL(%f,%f,%f,%f)" % (q[0], q[1], Z_HOVER, A.r))
d.sendRecvMsg("SetCollisionLevel(3)")
d.SpeedFactor(mover.TRAVEL_SPEED)
A.go_park()
print("parked", [round(v, 1) for v in a.pose()[:4]])
d.DisableRobot()
a.close()

import sys, os, time
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1"
sys.path.insert(0, os.path.join(R, "MG400")); os.chdir(os.path.join(R, "MG400"))
from dobot_api import DobotApiDashboard
DobotApiDashboard.log = lambda self, t: None
d = DobotApiDashboard("192.168.2.7", 29999)
m = lambda: d.sendRecvMsg("RobotMode()").strip()
print("mode", m(), "pose", d.sendRecvMsg("GetPose()").strip()[:60])
if "{6}" in m():
    print("StopDrag ->", d.sendRecvMsg("StopDrag()").strip()); time.sleep(1.0)
    print("mode", m())
print("err", d.sendRecvMsg("GetErrorID()").replace("\n", "").replace("\t", "")[:70])
d.close()

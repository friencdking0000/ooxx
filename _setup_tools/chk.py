import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1"
for d in ("MG400", "Calib"): sys.path.insert(0, os.path.join(R, d))
os.chdir(os.path.join(R, "MG400"))
import socket
from dobot_api import DobotApiDashboard
DobotApiDashboard.log = lambda self, t: None
for name, ip in (("A", "192.168.1.6"), ("B", "192.168.2.7")):
    try:
        s = socket.create_connection((ip, 29999), timeout=3); s.close()
    except Exception as e:
        print(name, ip, "連不上：", e); continue
    d = DobotApiDashboard(ip, 29999)
    try:
        print(name, ip, "mode", d.sendRecvMsg("RobotMode()").strip(), "| pose", d.sendRecvMsg("GetPose()").strip()[:70], "| err", d.sendRecvMsg("GetErrorID()").replace("\n", "").replace("\t", "")[:60])
    finally:
        d.close()
os.chdir(os.path.join(R, "Calib"))
import cv2, drift_check, find_pieces, ox_model
img = cv2.imread(os.path.join(R, "Arena", "cam_full.jpg"))
try:
    print("底座位移:", drift_check.base_shift(img))
except Exception as e:
    print("base_shift 失敗", e)
plan = json.load(open("cell_plan.json", encoding="utf-8"))
print("棋盤:", "".join(c if c != " " else "." for c in find_pieces.read_board(img, plan)))
print("模型偵測:", [(k, int(u), int(v), round(c, 2)) for u, v, r, k, c in ox_model.detect(img)])
